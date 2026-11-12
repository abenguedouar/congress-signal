"""Daily decision: what the portfolio should hold, and the orders that take it there.

    python scripts/trade.py              # dry run (default): writes the plan, sends nothing
    python scripts/trade.py --execute    # sends the orders to the Alpaca TEST account

The plan is written to data/store/live/plan_<date>.json in both cases. With
--execute the orders actually sent are appended to data/store/live/ledger.parquet.
Creating the file data/store/live/STOP blocks every order.
"""
import argparse
import json

import pandas as pd

from csig.config import ROOT, load_config
from csig.data.trades import signals
from csig.live.planner import plan_orders
from csig.live.target import target_weights


def main(execute: bool = False) -> dict:
    cfg = load_config()
    store, rules = ROOT / cfg["data"]["store_dir"], cfg["live"]
    live = store / "live"
    live.mkdir(parents=True, exist_ok=True)
    st = signals(pd.read_parquet(store / "trades.parquet"), cfg)
    prices = pd.read_parquet(store / "panel" / "prices_adj.parquet")
    calendar = prices[cfg["data"]["benchmark"]].dropna().index
    today = calendar[-1]                                   # latest day with prices: decisions use closes of that day
    priced = prices.columns[prices.loc[today].notna()]
    eligible = pd.read_parquet(store / "eligibility.parquet")
    target, active = target_weights(st, calendar, eligible, priced, today, cfg["portfolio"]["signal_ttl_days"])

    broker, notes = None, []
    try:
        from csig.live.alpaca import Broker
        broker = Broker()
        account, positions = broker.account(), broker.positions()
        if account["blocked"]:
            raise RuntimeError("the account is blocked")
        bad = [s for s in target.index if s not in positions and not broker.tradable(s)]
        if bad:
            notes.append(f"not tradable in fractions at the broker, left out: {', '.join(sorted(bad))}")
            target = target.drop(bad)
            target = target / target.sum() if len(target) else target
    except Exception as exc:                               # no keys or no connection: plan against an empty test account
        if execute:
            raise
        notes.append(f"broker not reached ({type(exc).__name__}: {exc}); plan computed for an empty account of 100,000")
        account, positions = {"equity": 100_000.0, "cash": 100_000.0, "status": "simulated"}, {}

    ledger_path = live / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path) if ledger_path.exists() else pd.DataFrame(
        columns=["date", "symbol", "side", "dollars", "qty", "reason", "order_id", "status"])
    last_buy = ledger[ledger["side"] == "buy"].groupby("symbol")["date"].max().to_dict()
    plan = plan_orders(target, positions, account["equity"], account["cash"], last_buy, today, calendar, rules)
    plan.update({"date": str(today.date()), "mode": "execute" if execute else "dry run", "account": account,
                 "holdings_before": len(positions), "target_names": int(len(target)),
                 "target": {k: round(float(v), 5) for k, v in target.items()},
                 "signals": int(len(active)), "members_behind": int(active["member"].nunique()) if len(active) else 0})
    plan["notes"] = notes + plan["notes"]

    if execute:
        if pd.Timestamp.now().normalize() < pd.Timestamp(str(rules["launch_date"])):
            plan["notes"].append(f"before the launch date ({rules['launch_date']}): no order sent")
        elif (live / "STOP").exists():
            plan["notes"].append("STOP file present: no order sent")
        elif plan["refused"]:
            plan["notes"].append("an order was refused by the safety rules: nothing sent, a person must look")
        else:
            sent = []
            for o in sorted(plan["orders"], key=lambda o: o["side"] != "sell"):      # sales first
                tag = f"{plan['date']}-{o['symbol']}-{o['side']}"                     # the broker rejects a repeated tag: one order per stock per day
                try:
                    r = broker.sell(o["symbol"], o["qty"], tag) if o["side"] == "sell" else broker.buy(o["symbol"], o["dollars"], tag)
                    sent.append({**o, "date": plan["date"], "order_id": r.get("id"), "status": r.get("status")})
                except Exception as exc:
                    plan["notes"].append(f"{o['symbol']} {o['side']}: not sent ({exc})")
            if sent:
                pd.concat([ledger, pd.DataFrame(sent)], ignore_index=True).to_parquet(ledger_path)
            plan["sent"] = len(sent)
    (live / f"plan_{plan['date']}.json").write_text(json.dumps(plan, indent=1))
    show = {k: plan[k] for k in ("date", "mode", "account", "holdings_before", "target_names", "signals", "members_behind", "summary", "refused", "notes")}
    print(json.dumps(show, indent=1))
    for o in plan["orders"][:60]:
        print(f"  {o['side']:4s} {o['symbol']:6s} {o['dollars']:>10,.2f}  {o['reason']}")
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="send the orders to the Alpaca test account")
    main(ap.parse_args().execute)
