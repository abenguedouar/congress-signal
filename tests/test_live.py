"""The live rule is the tested rule, and the order planner respects its safety rules."""
import numpy as np
import pandas as pd
import pytest

from csig.live.planner import plan_orders
from csig.live.target import WIDE, split_by_review, target_weights
from csig.portfolio import backtest

RULES = {"cash_reserve": 0.01, "min_order_dollars": 5.0, "no_trade_band": 0.2, "max_order_share": 0.15, "hold_days": 2}
DAYS = pd.bdate_range("2025-01-01", periods=200)


def test_live_target_is_the_backtest_portfolio():
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(30)]
    st = pd.DataFrame({"side": rng.choice(["buy", "sell"], 400, p=[0.7, 0.3]), "ticker": rng.choice(tickers, 400),
                       "disclosure_date": rng.choice(DAYS[:180] + pd.Timedelta(hours=0), 400), "member_name": "M"})
    returns = pd.DataFrame(rng.normal(0, 0.01, (len(DAYS), len(tickers))), index=DAYS, columns=tickers)
    returns.iloc[:60, :3] = np.nan                                     # three stocks without prices at the start
    eligible = pd.DataFrame(rng.random((4, len(tickers))) < 0.6, index=DAYS[[0, 50, 100, 150]], columns=tickers)
    buys = st[st.side == "buy"]
    i0 = np.searchsorted(DAYS.values, buys.disclosure_date.values, side="right")
    sig = pd.DataFrame({"t0": DAYS[np.minimum(i0, len(DAYS) - 1)], "ticker": buys.ticker.to_numpy(), "score": 1.0})[i0 < len(DAYS)]
    cfg = {**WIDE, "signal_ttl_days": 63, "rebalance_every_days": 5, "cost_bps": 10}
    book = backtest.run_backtest(sig, returns, cfg, eligible, "equal", 63)["book"]
    for day in list(book)[5::7]:
        live, _ = target_weights(st, DAYS, eligible, returns.columns[returns.loc[day].notna()], day, 63)
        assert sorted(live.index) == sorted(book[day].index)
        assert np.allclose(live.sort_index().to_numpy(), book[day].sort_index().to_numpy())
        assert live.sum() == pytest.approx(1.0)


def _plan(target, positions, equity=100_000.0, cash=100_000.0, last_buy=None, today=DAYS[50]):
    return plan_orders(pd.Series(target, dtype=float), positions, equity, cash, last_buy or {}, today, DAYS, RULES)


def test_first_day_buys_within_cash():
    plan = _plan({f"S{i}": 0.1 for i in range(10)}, {})
    assert plan["summary"]["buys"] == 10 and plan["summary"]["sells"] == 0
    assert plan["summary"]["bought"] == pytest.approx(99_000 * 0.1 * 10) and plan["summary"]["bought"] <= 100_000 * 0.99


def test_never_short_and_no_borrowing():
    positions = {"OLD": {"qty": 10.0, "value": 20_000.0}, "KEEP": {"qty": 5.0, "value": 18_000.0}}
    plan = _plan({"KEEP": 0.3, "NEW": 0.3, "NEW2": 0.4}, positions, equity=60_000.0, cash=10_000.0)
    sells = {o["symbol"]: o for o in plan["orders"] if o["side"] == "sell"}
    assert sells["OLD"]["qty"] == 10.0 and "KEEP" not in sells                 # sold in full, never more than held
    refused = {r["symbol"] for r in plan["refused"]}                           # 0.3 x 59,400 is above 15% of the account
    assert refused == {"NEW", "NEW2"} and plan["summary"]["bought"] == 0
    plan = _plan({f"N{i}": 0.1 for i in range(10)}, {"OLD": {"qty": 10.0, "value": 20_000.0}}, equity=60_000.0, cash=40_000.0)
    assert plan["summary"]["bought"] <= 40_000 + 20_000 - 600 + 0.01           # cash + sales - reserve


def test_purchase_not_sold_before_it_settled():
    positions = {"OLD": {"qty": 10.0, "value": 5_000.0}}
    assert _plan({}, positions, last_buy={"OLD": str(DAYS[49].date())})["orders"] == []
    assert _plan({}, positions, last_buy={"OLD": str(DAYS[48].date())})["orders"][0]["side"] == "sell"


def test_small_deviations_are_left_alone():
    target = {f"S{i}": 0.1 for i in range(10)}
    positions = {s: {"qty": 1.0, "value": 9_900.0 * (1.1 if s == "S0" else 0.9 if s == "S1" else 1.0)} for s in target}
    assert _plan(target, positions, cash=1_000.0)["orders"] == []
    positions["S0"]["value"] = 9_900.0 * 1.5
    orders = _plan(target, positions, cash=1_000.0)["orders"]
    assert [(o["symbol"], o["side"]) for o in orders] == [("S0", "sell"), ("S1", "buy")] and orders[0]["qty"] < 1.0
    assert orders[1]["dollars"] == pytest.approx(990.0)       # the cash freed by the trim goes to the holding under its target


def test_a_stock_that_leaves_the_target_is_sold_whole():
    days = pd.bdate_range("2026-11-10", periods=8)
    positions = {"AAA": {"qty": 9.021356234, "value": 2364.23}, "BBB": {"qty": 3.0, "value": 3000.0}}
    plan = plan_orders(pd.Series({"BBB": 0.5}), positions, 5400.0, 35.77, {"AAA": "2026-11-10"}, days[-1], days, {})
    (order,) = [o for o in plan["orders"] if o["symbol"] == "AAA"]
    assert order["side"] == "sell" and order["reason"] == "left the target" and order["qty"] == 9.021356234      # not rounded: nothing is left behind


def test_a_stock_waits_until_its_activity_is_checked():
    review = {"AAA": {"ok": True}, "BBB": {"ok": False, "why": "refused"}}
    assert split_by_review(["AAA", "BBB", "CCC"], review) == (["AAA"], ["CCC"], ["BBB"])


def test_idle_cash_is_put_back_to_work():
    target = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    positions = {s: {"qty": 1.0, "value": 2_400.0} for s in target}
    plan = _plan(target, positions, equity=8_700.0, cash=1_500.0)            # each holding is 16% under its target: inside the band
    assert plan["summary"]["buys"] == 3 and {o["reason"] for o in plan["orders"]} == {"below target"}
    assert 1_400 < plan["summary"]["bought"] <= 1_500 - 87                    # all the cash above the 1% reserve, never more
    assert _plan(target, positions, equity=7_400.0, cash=200.0)["orders"] == []      # under 3% of the account in cash: left alone
