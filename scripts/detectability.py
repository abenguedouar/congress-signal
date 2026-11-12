"""Detectability study: when can an informed legislator be told apart from a lucky one?

Outcomes are simulated; the structure (who trades, how often) is real.
Two experiments:

1. Phase transition. p members, k of them informed. For each pair
   (edge per purchase, purchases per member) the LASSO is fitted on simulated
   outcomes and we record how often its k largest coefficients are exactly
   the k informed members. Same logic as the compressed-sensing phase
   diagrams of the course: below a frontier recovery fails, above it succeeds.

2. The leaderboard illusion. With NO informed member at all, rank real
   members by the average outcome of their purchases and look at the top 5:
   how good do they look by luck alone?

    python scripts/detectability.py          # reads data/store/trades.parquet, writes to results/
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from csig.config import ROOT, load_config
from csig.data.trades import signals
from csig.sim import simulate_labels

warnings.filterwarnings("ignore")
NOISE_SD = 0.145     # st. dev. of the 63-day abnormal return of one disclosed purchase, measured on the real data


def lasso_member_coefs(who: np.ndarray, y: np.ndarray, p: int, seed: int) -> np.ndarray:
    X = np.zeros((len(who), p))
    X[np.arange(len(who)), who] = 1.0
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    order = np.random.default_rng(seed).permutation(len(y))       # outcomes are exchangeable here
    return LassoCV(cv=5, alphas=30, max_iter=5000).fit(X[order], y[order] - y.mean()).coef_


def phase_transition(p: int = 53, k: int = 3, edges=(0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14),
                     sizes=(5, 10, 20, 40, 80, 160), reps: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = pd.DataFrame(index=sizes, columns=edges, dtype=float)
    for n in sizes:
        counts = pd.Series(np.full(p, n))
        for e in edges:
            hits = 0
            for r in range(reps):
                who, y, informed = simulate_labels(counts, k, e, NOISE_SD, rng)
                coef = lasso_member_coefs(who, y, p, seed=r)
                top = np.argsort(-coef)[:k]
                hits += int(set(top) == set(np.flatnonzero(informed)) and (coef[top] > 0).all())
            grid.loc[n, e] = hits / reps
    return grid


def leaderboard_illusion(member_counts: pd.Series, top: int = 5, min_trades: int = 5, reps: int = 5000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    counts = member_counts[member_counts >= min_trades].to_numpy()
    best = np.empty(reps)
    for r in range(reps):
        means = rng.normal(0.0, NOISE_SD / np.sqrt(counts))       # average outcome of each member, no edge at all
        best[r] = np.sort(means)[-top:].mean()
    return {"members_ranked": int(len(counts)), "top": top, "noise_sd": NOISE_SD,
            "apparent_edge_of_top_mean": float(best.mean()),
            "apparent_edge_of_top_5pct_95pct": [float(np.quantile(best, 0.05)), float(np.quantile(best, 0.95))]}


def real_structure_power(member_counts: pd.Series, k: int = 3, edges=(0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20),
                         reps: int = 40, seed: int = 2) -> dict:
    """Same recovery test on the real distribution of purchases per member."""
    rng = np.random.default_rng(seed)
    counts = member_counts[member_counts >= 2].reset_index(drop=True)
    out = {}
    for e in edges:
        hits = 0
        for r in range(reps):
            who, y, informed = simulate_labels(counts, k, e, NOISE_SD, rng)
            coef = lasso_member_coefs(who, y, len(counts), seed=r)
            top = np.argsort(-coef)[:k]
            hits += int(set(top) == set(np.flatnonzero(informed)) and (coef[top] > 0).all())
        out[str(e)] = hits / reps
    return out


if __name__ == "__main__":
    cfg = load_config()
    out_dir = str(ROOT / "results")
    st = signals(pd.read_parquet(ROOT / cfg["data"]["store_dir"] / "trades.parquet"), cfg)
    lock = st["disclosure_date"].max() - pd.DateOffset(months=cfg["research"]["lockbox_months"])
    buys = st[(st["side"] == "buy") & (st["disclosure_date"] < lock)]      # development period only
    counts = buys["member_name"].value_counts()

    grid = phase_transition(p=100, reps=12)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(grid.columns)), [f"{int(e * 100)}%" for e in grid.columns])
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("True edge per purchase (abnormal return over 63 trading days)")
    ax.set_ylabel("Disclosed purchases per member")
    ax.set_title(f"Probability that the LASSO identifies the informed members\n"
                 f"(100 members, 3 informed, outcome noise {NOISE_SD * 100:.1f}%)", fontsize=10)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="recovery rate")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/detectability_phase_transition.png", dpi=160)

    res = {"noise_sd": NOISE_SD, "members": int(len(counts)), "purchases": int(counts.sum()),
           "median_purchases_per_member": float(counts.median()),
           "members_with_100_purchases": int((counts >= 100).sum()),
           "phase_transition": {str(n): {str(e): float(grid.loc[n, e]) for e in grid.columns} for n in grid.index},
           "real_structure_recovery_by_edge": real_structure_power(counts, reps=20),
           "leaderboard_illusion": leaderboard_illusion(counts)}
    with open(f"{out_dir}/detectability.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
