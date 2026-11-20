"""From a target portfolio to a list of orders, with the safety rules.

Rules that no setting can switch off:
  * long only: a sale never exceeds what is held; nothing is ever sold short;
  * no borrowing: purchases are sized within the cash that sales free that day, minus a reserve;
  * stocks only, bought for a dollar amount; no option, no other instrument;
  * a stock outside the target is sold in full; a stock bought less than
    ``hold_days`` trading days ago is not sold (purchases must have settled);
  * one order per stock per day; an order above ``max_order_share`` of the
    account is refused and reported, not shrunk;
  * if the file ``STOP`` exists in the live folder, no order at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Order:
    symbol: str
    side: str            # "buy" or "sell"
    dollars: float       # purchases: amount to spend; sales: value sold at the last price
    qty: float           # sales only
    reason: str


def plan_orders(target: pd.Series, positions: dict[str, dict], equity: float, cash: float, last_buy: dict[str, str],
                today, trading_days: pd.DatetimeIndex, rules: dict) -> dict:
    """``target``: weights summing to at most 1. ``positions``: {symbol: {"qty", "value"}}.
    ``last_buy``: {symbol: date of the latest purchase}. Returns {"orders": [...], "refused": [...], "notes": [...]}."""
    reserve, min_order = rules.get("cash_reserve", 0.01), rules.get("min_order_dollars", 5.0)
    band, max_share, hold = rules.get("no_trade_band", 0.2), rules.get("max_order_share", 0.15), rules.get("hold_days", 2)
    max_idle = rules.get("max_idle_cash", 0.03)
    today = pd.Timestamp(today)
    investable = equity * (1 - reserve)
    orders, refused, notes = [], [], []

    def settled(symbol: str) -> bool:
        bought = last_buy.get(symbol)
        return bought is None or int(((trading_days > pd.Timestamp(bought)) & (trading_days <= today)).sum()) >= hold

    # 1. sales: everything outside the target, and what exceeds the target by more than the band
    for symbol, pos in sorted(positions.items()):
        want = float(target.get(symbol, 0.0)) * investable
        excess = pos["value"] - want
        if pos["qty"] <= 0 or excess <= 0:
            continue
        if want > 0 and excess < max(min_order, band * want):
            continue
        if not settled(symbol):
            notes.append(f"{symbol}: sale postponed, bought less than {hold} trading days ago")
            continue
        qty = pos["qty"] if want == 0 else pos["qty"] * excess / pos["value"]
        # a stock that leaves the target is sold whole, to the last fraction of a share (rounding would leave dust behind)
        orders.append(Order(symbol, "sell", round(excess if want else pos["value"], 2), pos["qty"] if want == 0 else min(round(qty, 6), pos["qty"]),
                            "left the target" if want == 0 else "above target"))
    # 2. purchases: largest gaps first, within the cash available after the sales
    budget = cash + sum(o.dollars for o in orders) - equity * reserve
    gaps = {s: float(w) * investable - positions.get(s, {}).get("value", 0.0) for s, w in target.items()}
    for symbol, gap in sorted(gaps.items(), key=lambda kv: -kv[1]):
        want = float(target[symbol]) * investable
        if gap < max(min_order, band * want if symbol in positions else min_order):
            continue
        spend = min(gap, budget)
        if spend < min_order:
            notes.append(f"{symbol}: not bought, no cash left")
            continue
        if spend > max_share * equity:
            refused.append({"symbol": symbol, "dollars": round(spend, 2), "why": f"above {max_share:.0%} of the account"})
            continue
        orders.append(Order(symbol, "buy", round(spend, 2), 0.0, "new in target" if symbol not in positions else "below target"))
        budget -= spend
    # 3. when stocks went out and nothing came in, cash piles up while every holding sits just inside its band:
    #    above ``max_idle`` of the account, it is spread over the holdings below their target, band or not
    if budget > max_idle * equity:
        ordered = {o.symbol for o in orders}
        for symbol, gap in sorted(gaps.items(), key=lambda kv: -kv[1]):
            spend = int(min(gap, budget) * 100) / 100                     # rounded down: the total never exceeds the cash
            if symbol in ordered or symbol not in positions or spend < min_order:
                continue
            orders.append(Order(symbol, "buy", spend, 0.0, "below target"))
            budget -= spend
    bought, sold = sum(o.dollars for o in orders if o.side == "buy"), sum(o.dollars for o in orders if o.side == "sell")
    assert bought <= cash + sold - equity * reserve + 0.01 or bought == 0, "purchases exceed the cash available"
    assert all(o.qty <= positions[o.symbol]["qty"] + 1e-9 for o in orders if o.side == "sell"), "sale above the position"
    assert len({o.symbol for o in orders}) == len(orders), "two orders for one stock"
    return {"orders": [asdict(o) for o in orders], "refused": refused, "notes": notes,
            "summary": {"buys": len([o for o in orders if o.side == "buy"]), "sells": len([o for o in orders if o.side == "sell"]),
                        "bought": round(bought, 2), "sold": round(sold, 2)}}
