"""What the portfolio should hold on a given day.

The rule is the one fixed before launch and tested as "copy_everything|screened":
hold, in equal parts, every stock with a purchase disclosed during the last
``signal_ttl_days`` trading days, if the stock passes the mandate screen on
that day. It is computed with the functions of the backtest, so that what is
traded is what was tested (tests/test_live.py checks that the two coincide).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio.backtest import _eligible_on
from ..portfolio.optimizer import equal_weights

WIDE = {"max_names": 10 ** 6, "max_weight": 1.0, "min_weight": 0.0}


def purchase_signals(st: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per disclosed purchase: ticker, i0 (the first trading day strictly after the disclosure) and who bought."""
    buys = st[st["side"] == "buy"]
    i0 = np.searchsorted(calendar.values, buys["disclosure_date"].values.astype("datetime64[ns]"), side="right")
    col = lambda c: buys[c].to_numpy() if c in buys else np.full(len(buys), None, dtype=object)
    return pd.DataFrame({"ticker": buys["ticker"].to_numpy(), "i0": i0, "member": buys["member_name"].to_numpy(),
                         "disclosed": buys["disclosure_date"].to_numpy(), "chamber": col("chamber"), "traded": col("trade_date"),
                         "amount_min": col("amount_min"), "amount_max": col("amount_max"), "owner": col("owner")})


def split_by_review(candidates, review: dict) -> tuple[list[str], list[str], list[str]]:
    """(approved, waiting, refused). ``review``: {ticker: {"ok": bool, ...}}, the record of the activity checks that
    the private mandate asks for before a stock is bought for the first time. A stock without a record waits."""
    approved = [t for t in candidates if review.get(t, {}).get("ok") is True]
    waiting = [t for t in candidates if t not in review]
    refused = [t for t in candidates if t in review and review[t].get("ok") is not True]
    return approved, waiting, refused


def target_weights(st: pd.DataFrame, calendar: pd.DatetimeIndex, eligible: pd.DataFrame, priced, today,
                   ttl_days: int = 63) -> tuple[pd.Series, pd.DataFrame]:
    """(weights that sum to 1, the active signals behind them) as of the close of ``today``."""
    past = calendar[calendar <= pd.Timestamp(today)]
    if not len(past):
        return pd.Series(dtype=float), pd.DataFrame()
    i = len(past) - 1
    sig = purchase_signals(st, calendar)
    active = sig[(sig["i0"] <= i) & (sig["i0"] > i - ttl_days)]
    scores = active.groupby("ticker").size().astype(float)
    ok = _eligible_on(eligible, past[-1], scores.index)
    scores = scores[scores.index.isin(ok) & scores.index.isin(pd.Index(priced))]
    weights = equal_weights(scores, WIDE)
    weights = weights[weights > 0]
    return weights, active[active["ticker"].isin(weights.index)]
