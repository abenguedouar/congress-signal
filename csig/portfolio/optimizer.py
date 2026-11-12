"""Target weights from scores: a convex program.

    maximise   mu'w  -  gamma * w' Sigma w  -  kappa * ||w - w_prev||_1
    subject to w >= 0,  sum(w) <= 1,  w_i <= w_max

* mu      expected abnormal return of each candidate (annualised model score)
* Sigma   shrunk covariance of daily returns (annualised)
* the l1 term is a convex stand-in for "trade as few lines as possible":
  it keeps most weights exactly where they were, which is what limits costs
  on a small account.

The portfolio is long only and may hold cash: with few good candidates the
budget is not forced into them.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrunk_cov(returns: pd.DataFrame) -> np.ndarray:
    r = returns.fillna(0.0).to_numpy()
    if r.shape[0] < 20:
        return np.diag(np.full(r.shape[1], 0.3 ** 2))           # prior: 30% annual volatility
    return LedoitWolf().fit(r).covariance_ * 252.0


def target_weights(mu: pd.Series, cov: np.ndarray, w_prev: pd.Series | None, cfg: dict) -> pd.Series:
    """``mu`` is indexed by ticker; returns weights on the same index (may sum to less than 1)."""
    n = len(mu)
    if n == 0:
        return pd.Series(dtype=float)
    prev = np.zeros(n) if w_prev is None else w_prev.reindex(mu.index).fillna(0.0).to_numpy()
    w = cp.Variable(n)
    objective = (mu.to_numpy() @ w - cfg["risk_aversion"] * cp.quad_form(w, cp.psd_wrap(cov))
                 - cfg["turnover_penalty"] * cp.norm1(w - prev))
    problem = cp.Problem(cp.Maximize(objective), [w >= 0, cp.sum(w) <= 1, w <= cfg["max_weight"]])
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver)
            if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.SolverError:
            continue
    if w.value is None:                                          # never trade on a failed solve
        return pd.Series(prev, index=mu.index)
    out = pd.Series(np.clip(w.value, 0.0, cfg["max_weight"]), index=mu.index)
    out[out < cfg["min_weight"]] = 0.0                           # no dust positions
    if (out > 0).sum() > cfg["max_names"]:
        out[out.rank(ascending=False, method="first") > cfg["max_names"]] = 0.0
    return out


def equal_weights(scores: pd.Series, cfg: dict) -> pd.Series:
    """Baseline sizing: the best ``max_names`` positive scores, equally weighted, capped."""
    top = scores[scores > 0].nlargest(cfg["max_names"])
    out = pd.Series(0.0, index=scores.index)
    if len(top):
        out[top.index] = min(cfg["max_weight"], 1.0 / len(top))
    return out
