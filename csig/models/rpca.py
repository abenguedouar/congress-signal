"""Robust PCA: separating herd trades from unusual ones.

The member x stock matrix M of disclosed purchases is modelled as

    M = L + S,   L low rank (what everybody buys: a few common "taste" factors),
                 S sparse   (a handful of entries that do not fit the common pattern).

Minimising rank(L) + lambda * ||S||_0 is NP-hard. Replacing the rank by the
nuclear norm and l0 by l1 gives Principal Component Pursuit, a convex problem:

    minimise ||L||_* + lambda ||S||_1   subject to   L + S = M.

It is solved by an augmented Lagrangian scheme whose two steps are exactly
the proximal operators of the course: soft-thresholding of the singular
values (prox of the nuclear norm) and entrywise soft-thresholding (prox of l1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def soft(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svt(x: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular value thresholding: prox of tau * nuclear norm."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    rank = int((s > 0).sum())
    return (u[:, :rank] * s[:rank]) @ vt[:rank], rank


def pcp(M: np.ndarray, lam: float | None = None, tol: float = 1e-7, max_iter: int = 1000) -> dict:
    """Principal Component Pursuit by the inexact augmented Lagrangian method."""
    M = np.asarray(M, dtype=float)
    n1, n2 = M.shape
    lam = lam if lam is not None else 1.0 / np.sqrt(max(n1, n2))
    norm_m = np.linalg.norm(M)
    if norm_m == 0:
        return {"L": M.copy(), "S": M.copy(), "rank": 0, "iterations": 0, "lam": lam}
    spec = np.linalg.norm(M, 2)
    Y = M / max(spec, np.abs(M).max() / lam)
    mu, mu_max, rho = 1.25 / spec, 1.25 / spec * 1e7, 1.5
    S = np.zeros_like(M)
    for it in range(1, max_iter + 1):
        L, rank = svt(M - S + Y / mu, 1.0 / mu)
        S = soft(M - L + Y / mu, lam / mu)
        R = M - L - S
        Y = Y + mu * R
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(R) / norm_m < tol:
            break
    return {"L": L, "S": S, "rank": rank, "iterations": it, "lam": lam}


def trade_matrix(trades: pd.DataFrame, min_members: int = 2, min_tickers: int = 2) -> pd.DataFrame:
    """Members x tickers: log10 of the dollar amount bought (0 when no purchase).

    Rows and columns with a single non-zero entry carry no information about
    common behaviour and are dropped.
    """
    buys = trades[trades["side"] == "buy"]
    key = buys["member_id"].fillna(buys["member_name"])
    m = buys.assign(member_key=key).pivot_table(index="member_key", columns="ticker", values="amount_mid",
                                                 aggfunc="sum", fill_value=0.0)
    m = np.log10(m.clip(lower=1.0))
    for _ in range(5):
        m = m.loc[(m > 0).sum(axis=1) >= min_tickers, (m > 0).sum(axis=0) >= min_members]
    return m


def unusual_scores(trades: pd.DataFrame, lam_scale: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Share of each purchase that the common pattern fails to explain, in [0, 1].

    0 means the purchase is fully explained by the low-rank part (herd),
    1 means it sits entirely in the sparse part (unusual for that member and stock).
    """
    M = trade_matrix(trades)
    lam = lam_scale / np.sqrt(max(M.shape))
    fit = pcp(M.to_numpy(), lam)
    L, S = fit["L"], fit["S"]
    share = np.abs(S) / (np.abs(S) + np.abs(L) + 1e-12)
    rows, cols = np.nonzero(M.to_numpy())
    out = pd.DataFrame({"member_key": M.index[rows], "ticker": M.columns[cols],
                        "bought": M.to_numpy()[rows, cols], "herd_part": L[rows, cols],
                        "unusual_part": S[rows, cols], "unusual_share": share[rows, cols]})
    info = {"members": M.shape[0], "tickers": M.shape[1], "rank": fit["rank"], "lam": lam,
            "sparse_entries": int((np.abs(S) > 1e-8).sum()), "iterations": fit["iterations"],
            "nonzero_entries": int(len(rows))}
    return out, info


def rolling_unusual(events: pd.DataFrame, trades: pd.DataFrame, window_days: int = 365, lam_scale: float = 1.5,
                    freq: str = "QE") -> pd.Series:
    """Unusual score of each NEW purchase, using only purchases disclosed before it.

    At each period end q, the herd pattern L is estimated on the purchases
    disclosed during the previous ``window_days``. A purchase disclosed in the
    following period is scored 1 - (what the herd pattern predicted for that
    member and stock) / (what was actually bought), clipped to [0, 1]:
    0 = exactly what this member's usual pattern predicted, 1 = out of pattern.
    """
    out = pd.Series(np.nan, index=events.index)
    key = trades["member_id"].fillna(trades["member_name"])
    trades = trades.assign(member_key=key)
    ends = pd.date_range(events["disclosure_date"].min().normalize(), events["disclosure_date"].max() + pd.offsets.QuarterEnd(1), freq=freq)
    for q0, q1 in zip([None] + list(ends[:-1]), ends):
        cur = events[(events["disclosure_date"] <= q1) & ((events["disclosure_date"] > q0) if q0 is not None else True)]
        if cur.empty or q0 is None:
            continue
        past = trades[(trades["disclosure_date"] <= q0) & (trades["disclosure_date"] > q0 - pd.Timedelta(days=window_days))]
        M = trade_matrix(past) if len(past) else pd.DataFrame()
        if M.shape[0] < 5 or M.shape[1] < 5:
            continue
        L = pd.DataFrame(pcp(M.to_numpy(), lam_scale / np.sqrt(max(M.shape)))["L"], index=M.index, columns=M.columns)
        for idx, r in cur.iterrows():
            predicted = L.at[r["member_key"], r["ticker"]] if (r["member_key"] in L.index and r["ticker"] in L.columns) else 0.0
            out[idx] = 1.0 - float(np.clip(predicted / max(np.log10(max(r["amount_mid"], 10.0)), 1e-9), 0.0, 1.0))
    return out
