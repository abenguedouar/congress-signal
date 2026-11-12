"""Daily backtest with costs, and the random-portfolio benchmark.

Order of events on trading day d:
  1. the close of d is observed; signals with t0 <= d are known;
  2. on rebalance days, target weights are computed and traded at that close;
     the trade pays ``cost_bps`` per unit of turnover;
  3. the portfolio earns the return of day d+1 on the new weights.
A signal is used from its t0 (first trading day after disclosure) and for
``signal_ttl_days`` trading days. Uninvested money sits in cash at 0%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import equal_weights, shrunk_cov, target_weights


def _eligible_on(eligible: pd.DataFrame | None, date, tickers) -> pd.Index:
    if eligible is None:
        return pd.Index(tickers)
    past = eligible.index[eligible.index <= date]
    if len(past) == 0:
        return pd.Index([])
    row = eligible.loc[past[-1]].reindex(tickers).fillna(False)
    return row.index[row.to_numpy(dtype=bool)]


def run_backtest(signals: pd.DataFrame, returns: pd.DataFrame, cfg: dict, eligible: pd.DataFrame | None = None,
                 sizing: str = "optimizer", horizon_days: int = 63, vetoes: pd.DataFrame | None = None) -> dict:
    """``signals``: columns t0, ticker, score.  ``returns``: daily simple returns, date x ticker.

    ``vetoes`` (columns t0, ticker): disclosed sales. While one is active
    (``veto_days`` trading days from its t0) the stock is neither bought nor kept.
    """
    cal = returns.index
    veto = None
    if vetoes is not None and len(vetoes):
        veto = vetoes.assign(i0=cal.searchsorted(vetoes["t0"]))[["i0", "ticker"]]
    veto_days = cfg.get("veto_days", 21)
    sig = signals.dropna(subset=["score"]).copy()
    sig["i0"] = cal.searchsorted(sig["t0"])
    sig = sig[sig["i0"] < len(cal)]
    ttl, every, cost = cfg["signal_ttl_days"], cfg["rebalance_every_days"], cfg["cost_bps"] / 1e4
    start = int(sig["i0"].min()) if len(sig) else len(cal)

    w = pd.Series(dtype=float)
    daily, book = [], {}
    for i in range(start, len(cal) - 1):
        turnover = 0.0
        if (i - start) % every == 0:
            active = sig[(sig["i0"] <= i) & (sig["i0"] > i - ttl)]
            by_ticker = active.groupby("ticker")
            scores = by_ticker["score"].sum()
            scores = scores[scores > 0]
            scores = scores + 1e-9 * (by_ticker["i0"].max().reindex(scores.index) - i + ttl)   # equal scores: latest disclosure first
            ok = _eligible_on(eligible, cal[i], scores.index)
            priced = returns.columns[returns.iloc[i].notna()]
            scores = scores[scores.index.isin(ok) & scores.index.isin(priced)]
            if veto is not None:
                blocked = veto.loc[(veto["i0"] <= i) & (veto["i0"] > i - veto_days), "ticker"]
                scores = scores[~scores.index.isin(blocked)]
            if sizing == "optimizer" and len(scores):
                hist = returns.iloc[max(0, i - 252):i + 1][scores.index]
                target = target_weights(scores * 252.0 / horizon_days, shrunk_cov(hist), w, cfg)
            else:
                target = equal_weights(scores, cfg)
            target = target[target > 0]
            names = w.index.union(target.index)
            turnover = float((target.reindex(names).fillna(0) - w.reindex(names).fillna(0)).abs().sum())
            w = target
            book[cal[i]] = w.copy()
        r_next = returns.iloc[i + 1].reindex(w.index).fillna(0.0)
        gross = float((w * r_next).sum())
        daily.append({"date": cal[i + 1], "ret": gross - turnover * cost, "gross": gross,
                      "turnover": turnover, "n_names": int(len(w)), "exposure": float(w.sum())})
        if len(w):                                               # weights drift with prices
            grown = w * (1 + r_next)
            w = grown / (1 + gross)
    out = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame(
        columns=["ret", "gross", "turnover", "n_names", "exposure"])
    return {"daily": out, "book": book}


def random_portfolios(strategy: dict, returns: pd.DataFrame, cfg: dict, eligible: pd.DataFrame | None = None,
                      n_sims: int = 200, seed: int = 0, size: pd.DataFrame | None = None, n_buckets: int = 3) -> pd.DataFrame:
    """Null distribution: the same book with the names replaced at random.

    Each stock the strategy buys is swapped for a random stock of the eligible
    universe, bought on the same day with the same weight and held exactly as
    long. Timing, number of lines, exposure, turnover and costs are therefore
    identical; only the choice of names differs. A strategy that looks good
    merely because its universe did well, or because it was invested at the
    right time, sits in the middle of this distribution.

    With ``size`` (market value, date x ticker) the stand-in is drawn among the
    eligible stocks of the same size group on that day. Without it, a book of
    large companies is compared with mostly small ones, whose returns differ
    for reasons that have nothing to do with the choice being tested.
    """
    rng = np.random.default_rng(seed)
    cal, book = returns.index, strategy["book"]
    dates = sorted(book)
    cost = cfg["cost_bps"] / 1e4
    days = strategy["daily"].index
    R = returns.to_numpy(dtype=float)
    valid = ~np.isnan(R)
    R = np.nan_to_num(R, nan=0.0)
    col = {t: k for k, t in enumerate(returns.columns)}
    first_day = cal.get_loc(days[0]) if len(days) else 0
    pos = [cal.get_loc(d) for d in dates]
    pools, groups, group_of = [], [], []          # per rebalance: all candidates, candidates by size group, group of each name
    for d, i in zip(dates, pos):
        names = returns.columns[valid[i]]
        names = names[names.isin(_eligible_on(eligible, d, names))]
        pools.append(np.array([col[t] for t in names], dtype=int))
        by_group, of = {}, {}
        if size is not None and len(names):
            past = size.index[size.index <= d]
            if len(past):
                caps = size.loc[past[-1]]
                known = caps.reindex(names).dropna()
                if len(known) >= 3 * n_buckets:
                    edges = np.quantile(known.to_numpy(), np.linspace(0, 1, n_buckets + 1)[1:-1])
                    which = np.searchsorted(edges, known.to_numpy(), side="right")
                    by_group = {b: np.array([col[t] for t in known.index[which == b]], dtype=int) for b in range(n_buckets)}
                    mine = caps.reindex(book[d].index).dropna()
                    of = dict(zip(mine.index, np.searchsorted(edges, mine.to_numpy(), side="right").tolist()))
        groups.append(by_group)
        group_of.append(of)
    sims = np.zeros((n_sims, len(days)))
    for s in range(n_sims):
        twin: dict[str, int] = {}                                # strategy name -> column of its random stand-in
        held = np.array([], dtype=int)
        w = np.array([])
        for j, d in enumerate(dates):
            i = pos[j]
            end = pos[j + 1] if j + 1 < len(dates) else len(cal) - 1
            target = book[d]
            twin = {k: v for k, v in twin.items() if k in target.index}
            for name in target.index:
                if name not in twin:
                    used = np.fromiter(twin.values(), dtype=int, count=len(twin))
                    free = np.setdiff1d(groups[j].get(group_of[j].get(name, -1), pools[j]), used)
                    if not len(free):                             # size group exhausted: any eligible stock
                        free = np.setdiff1d(pools[j], used)
                    if len(free):
                        twin[name] = int(rng.choice(free))
            keys = [k for k in target.index if k in twin]
            new_cols = np.array([twin[k] for k in keys], dtype=int)
            new_w = target.reindex(keys).to_numpy(dtype=float)
            before = dict(zip(held.tolist(), w.tolist()))
            after = dict(zip(new_cols.tolist(), new_w.tolist()))
            turnover = sum(abs(after.get(c, 0.0) - before.get(c, 0.0)) for c in set(before) | set(after))
            held, w = new_cols, new_w
            for k in range(i + 1, end + 1):
                r = R[k, held]
                gross = float(w @ r) if len(w) else 0.0
                if 0 <= k - first_day < sims.shape[1]:
                    sims[s, k - first_day] = gross - (turnover * cost if k == i + 1 else 0.0)
                if len(w):
                    w = w * (1 + r) / (1 + gross)
    return pd.DataFrame(sims, columns=days)
