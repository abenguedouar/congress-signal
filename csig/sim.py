"""Synthetic market with a planted edge.

Used for two things only:
* tests: the pipeline must find an edge that was planted, and must find
  nothing when none was planted;
* the detectability study: how large must an edge be, and how many trades are
  needed, before it can be told apart from luck?

Nothing produced here is evidence about real legislators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_market(n_days: int = 1500, n_tickers: int = 120, n_members: int = 60, n_informed: int = 4,
                     trades_per_member: int = 40, edge: float = 0.06, horizon: int = 63,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Returns (trades, prices, informed_member_ids).

    Prices follow a one-factor model. After each purchase disclosed by an
    informed member, the stock earns ``edge`` in excess of the market, spread
    evenly over the next ``horizon`` trading days.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-02", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_tickers)
    rets = beta[None, :] * market[:, None] + rng.normal(0, 0.018, (n_days, n_tickers))
    tickers = [f"T{k:03d}" for k in range(n_tickers)]
    members = [f"M{k:03d}" for k in range(n_members)]
    informed = members[:n_informed]

    rows = []
    for m in members:
        for j in range(trades_per_member):
            i_trade = int(rng.integers(5, n_days - horizon - 60))
            lag = int(rng.integers(3, 40))
            i_disc = min(i_trade + lag, n_days - horizon - 5)
            k = int(rng.integers(0, n_tickers))
            rows.append({"trade_id": f"{m}-{j}", "chamber": "house", "member_id": m, "member_name": m,
                         "owner": "self", "ticker": tickers[k], "asset_name": tickers[k], "asset_type": "stock",
                         "side": "buy", "trade_date": cal[i_trade], "disclosure_date": cal[i_disc],
                         "amount_min": 1001.0, "amount_max": 15000.0, "source_url": f"{m}-{j // 3}"})
            if m in informed:
                rets[i_disc + 2:i_disc + 2 + horizon, k] += edge / horizon   # starts the day after t0
    trades = pd.DataFrame(rows).sort_values("disclosure_date").reset_index(drop=True)
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=cal, columns=tickers)
    prices["SPY"] = 100 * np.cumprod(1 + market)
    return trades, prices, informed


def simulate_labels(member_counts: pd.Series, n_informed: int, edge: float, noise_sd: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-level simulation on a real trade structure.

    ``member_counts`` is the number of disclosed purchases of each real member.
    ``n_informed`` members, drawn at random, earn ``edge`` on every purchase;
    all outcomes carry noise of standard deviation ``noise_sd``.
    Returns (member index of each event, outcomes, true informed mask).
    """
    counts = member_counts.to_numpy()
    who = np.repeat(np.arange(len(counts)), counts)
    informed = np.zeros(len(counts), dtype=bool)
    informed[rng.choice(len(counts), size=n_informed, replace=False)] = True
    y = rng.normal(0.0, noise_sd, len(who)) + edge * informed[who]
    return who, y, informed
