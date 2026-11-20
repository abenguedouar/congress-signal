"""Daily decision: what the portfolio should hold, and the orders that take it there.

    python scripts/trade.py              # dry run (default): writes the plan, sends nothing
    python scripts/trade.py --execute    # sends the orders to the Alpaca TEST account

The plan is written to data/store/live/plan_<date>.json in both cases. With
--execute the orders actually sent are appended to data/store/live/ledger.parquet.
Creating the file data/store/live/STOP blocks every order.
"""
import argparse
import json
import time

import pandas as pd

from csig.config import ROOT, load_config
from csig.data.trades import signals
from csig.live.context import build_context
from csig.live.planner import plan_orders
from csig.live.target import split_by_review, target_weights


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

    broker, notes, waiting = None, [], []
    review_path = live / "activity_review.json"            # the mandate's activity check: a stock without a record is not bought yet
    if review_path.exists():
        review = json.loads(review_path.read_text())
        approved, waiting, refused = split_by_review(list(target.index), review)
        behind = active[active["ticker"].isin(waiting)]
        if waiting or refused:
            target, active = target_weights(st, calendar, eligible, pd.Index([t for t in priced if review.get(t, {}).get("ok") is True]),
                                            today, cfg["portfolio"]["signal_ttl_days"])
            if waiting:
                notes.append(f"waiting for the activity check, not bought: {', '.join(sorted(waiting))}")
            if refused:
                notes.append(f"refused by the activity check: {', '.join(sorted(refused))}")
        (live / "pending_review.json").write_text(json.dumps(
            [{"ticker": t, "members": sorted(behind[behind["ticker"] == t]["member"].unique())} for t in sorted(waiting)] if waiting else []))
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
        elif not broker.clock().get("is_open"):
            plan["notes"].append("the market is closed: no order sent")
        else:
            if account.get("shorting_enabled") or account.get("multiplier", 1) > 1:     # the account itself refuses to borrow or sell short
                try:
                    broker.configure(no_shorting=True, max_margin_multiplier="1")
                    plan["notes"].append("account set to: no short selling, no borrowing")
                except Exception as exc:
                    plan["notes"].append(f"account settings not changed ({exc}); the planner never borrows or sells short anyway")
            sent, waited = [], False
            for o in sorted(plan["orders"], key=lambda o: o["side"] != "sell"):      # sales first
                if o["side"] == "buy" and sent and not waited:                       # purchases paid by today's sales: wait until these are done
                    waited = True
                    for _ in range(30):
                        if not broker.open_orders():
                            break
                        time.sleep(2)
                tag = f"{plan['date']}-{o['symbol']}-{o['side']}"                     # the broker rejects a repeated tag: one order per stock per day
                try:
                    if o["side"] == "sell":
                        r = broker.close(o["symbol"]) if o["reason"] == "left the target" else broker.sell(o["symbol"], o["qty"], tag)
                    else:
                        r = broker.buy(o["symbol"], o["dollars"], tag)
                    sent.append({**o, "date": plan["date"], "order_id": r.get("id"), "status": r.get("status")})
                except Exception as exc:
                    plan["notes"].append(f"{o['symbol']} {o['side']}: not sent ({exc})")
            if sent:
                ledger = pd.concat([ledger, pd.DataFrame(sent)], ignore_index=True)
                ledger.to_parquet(ledger_path)
            plan["sent"] = len(sent)
            stamp = pd.Timestamp.now(tz="UTC")                  # today's decision is made: later scheduled starts stop at once
            (live / "last_run.json").write_text(json.dumps({"utc_date": str(stamp.date()), "at": stamp.strftime("%H:%M UTC"), "sent": len(sent)}))
    (live / f"plan_{plan['date']}.json").write_text(json.dumps(plan, indent=1))
    context = build_context(cfg, store, sorted(set(target.index) | set(positions)), active, ledger)   # what the page shows beside the account
    (live / "context.json").write_text(json.dumps(context))
    show = {k: plan[k] for k in ("date", "mode", "account", "holdings_before", "target_names", "signals", "members_behind", "summary", "refused", "notes")}
    print(json.dumps(show, indent=1))
    for o in plan["orders"][:60]:
        print(f"  {o['side']:4s} {o['symbol']:6s} {o['dollars']:>10,.2f}  {o['reason']}")
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="send the orders to the Alpaca test account")
    main(ap.parse_args().execute)
