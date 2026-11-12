"""Performance statistics. All inputs are daily simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def summary(ret: pd.Series, bench: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {"days": int(len(ret))}
    equity = (1 + ret).cumprod()
    years = len(ret) / DAYS
    out = {
        "days": int(len(ret)),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "volatility": float(ret.std(ddof=1) * np.sqrt(DAYS)),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(DAYS)) if ret.std(ddof=1) > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }
    if bench is not None:
        b = bench.reindex(ret.index).fillna(0.0)
        beta = float(np.cov(ret, b, ddof=1)[0, 1] / b.var(ddof=1)) if b.var(ddof=1) > 0 else 0.0
        active = ret - b
        out.update({
            "benchmark_total_return": float((1 + b).prod() - 1),
            "beta": beta,
            "alpha_annual": float((ret.mean() - beta * b.mean()) * DAYS),
            "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(DAYS)) if active.std(ddof=1) > 0 else 0.0,
        })
    return out


def alpha_interval(ret: pd.Series, bench: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 0) -> dict:
    """Block-bootstrap confidence interval for the annual alpha.

    Daily returns are autocorrelated in volatility; resampling blocks of about
    one month keeps that dependence instead of pretending days are independent.
    """
    df = pd.concat([ret, bench.reindex(ret.index)], axis=1).dropna()
    x = df.to_numpy()
    n = len(x)
    if n < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    alphas = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        r, b = x[idx, 0], x[idx, 1]
        beta = np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1)
        alphas[k] = (r.mean() - beta * b.mean()) * DAYS
    lo, hi = np.quantile(alphas, [0.025, 0.975])
    return {"alpha_ci_low": float(lo), "alpha_ci_high": float(hi), "prob_alpha_positive": float((alphas > 0).mean())}


def null_pvalue(strategy_ret: pd.Series, null_returns: pd.DataFrame) -> dict:
    """Where does the strategy's total return sit among the random portfolios?"""
    s = float((1 + strategy_ret).prod() - 1)
    null = (1 + null_returns).prod(axis=1) - 1
    return {"strategy_total": s, "null_median": float(null.median()),
            "null_5pct": float(null.quantile(0.05)), "null_95pct": float(null.quantile(0.95)),
            "p_value": float((null >= s).mean()), "null_totals": [round(float(x), 4) for x in null]}


def periodic_table(ret: pd.Series, bench: pd.Series | None = None, freq: str = "ME") -> pd.DataFrame:
    """Monthly ('ME'), quarterly ('QE') or annual ('YE') returns for the public reports."""
    out = pd.DataFrame({"portfolio": (1 + ret).resample(freq).prod() - 1})
    if bench is not None:
        out["benchmark"] = (1 + bench.reindex(ret.index).fillna(0.0)).resample(freq).prod() - 1
        out["difference"] = out["portfolio"] - out["benchmark"]
    return out


def factor_alpha(ret: pd.Series, factors: pd.DataFrame) -> dict:
    """Alpha left after known factors are removed (market, size, value, profitability, investment, momentum).

    ``factors`` holds daily factor returns and a risk-free column ``RF``, in decimals.
    A strategy whose gain is only a tilt towards, say, large profitable growth
    stocks shows a large raw excess return and an alpha near zero here.
    Standard errors are Newey-West (21 lags) because daily returns are serially dependent.
    """
    df = pd.concat([ret.rename("ret"), factors], axis=1).dropna()
    if len(df) < 120:
        return {}
    cols = [c for c in df.columns if c not in ("ret", "RF")]
    y = (df["ret"] - df.get("RF", 0.0)).to_numpy()
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    lags = 21
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        g = (X[k:] * resid[k:, None]).T @ (X[:-k] * resid[:-k, None])
        S += (1 - k / (lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ S @ xtx_inv))
    out = {"factor_alpha_annual": float(beta[0] * DAYS), "factor_alpha_tstat": float(beta[0] / se[0])}
    out.update({f"loading_{c}": float(b) for c, b in zip(cols, beta[1:])})
    return out


def deflated_sharpe(ret: pd.Series, trial_sharpes: list[float]) -> dict:
    """Probability that the Sharpe ratio is positive once the number of rules tried is accounted for.

    Trying many rules and keeping the best one inflates its Sharpe ratio even
    when none has any merit. The deflated Sharpe ratio (Bailey and Lopez de
    Prado, 2014) compares the observed ratio with the best one expected by
    luck among ``len(trial_sharpes)`` unskilled rules, and corrects for
    non-normal returns. Values near 1 are reassuring; near 0.5 or below, the
    result is what luck alone would produce.
    """
    from scipy.stats import kurtosis, norm, skew
    r = ret.dropna().to_numpy()
    n, trials = len(r), [x for x in trial_sharpes if np.isfinite(x)]
    if n < 60 or len(trials) < 2 or r.std(ddof=1) == 0:
        return {}
    sr = r.mean() / r.std(ddof=1)                                # per day
    var_trials = np.var(np.array(trials) / np.sqrt(DAYS), ddof=1)
    N, euler = len(trials), 0.5772156649
    sr0 = np.sqrt(var_trials) * ((1 - euler) * norm.ppf(1 - 1 / N) + euler * norm.ppf(1 - 1 / (N * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return {"deflated_sharpe": float(norm.cdf(z)), "trials": N,
            "sharpe_expected_from_luck_annual": float(sr0 * np.sqrt(DAYS))}
