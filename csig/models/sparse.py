"""Which few legislators and trade characteristics predict post-disclosure returns?

The model is linear: abnormal return = sum over features of beta_j x_j + noise,
with one indicator per active member plus the trade characteristics. Hundreds
of coefficients, a few thousand noisy observations, and the belief that most
coefficients are zero: the high-dimensional sparse regression of the course.

* Best-subset selection (penalise the number of non-zero coefficients, the
  "l0 norm") is NP-hard.
* Its convex relaxation replaces l0 by the l1 norm: the LASSO.

Everything is evaluated walk-forward: a model fitted on outcomes fully known
before date d predicts the events disclosed after d. Outcomes overlap in
time, so the training set stops where the last training outcome was observed
("purging"); otherwise the test period leaks into the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, OrthogonalMatchingPursuitCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.events import FEATURES


def design_matrix(ev: pd.DataFrame, min_member_trades: int = 8, features: list[str] | None = None) -> pd.DataFrame:
    """Trade characteristics plus one 0/1 column per member with enough disclosed purchases."""
    features = features or FEATURES
    X = ev[features].astype(float).copy()
    counts = ev["member_key"].value_counts()
    for m in counts[counts >= min_member_trades].index:
        X[f"member::{m}"] = ev["member_key"].eq(m).astype(float)
    return X.fillna(0.0)


def make_model(kind: str, cv_folds: int = 5, seed: int = 0):
    cv = TimeSeriesSplit(n_splits=cv_folds)
    if kind == "lasso":
        return make_pipeline(StandardScaler(), LassoCV(cv=cv, alphas=40, max_iter=20000, random_state=seed))
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
    if kind == "ols":
        return make_pipeline(StandardScaler(), LinearRegression())
    if kind == "omp":   # greedy l0 heuristic, the non-convex competitor of the LASSO
        return make_pipeline(StandardScaler(), OrthogonalMatchingPursuitCV(cv=cv))
    if kind == "gbm":   # flexible non-linear benchmark
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    raise ValueError(kind)


def coefficients(model, columns) -> pd.Series:
    """Coefficients on the standardised scale (zero when not available, e.g. boosting)."""
    last = model[-1] if hasattr(model, "__getitem__") else model
    coef = getattr(last, "coef_", None)
    return pd.Series(np.zeros(len(columns)) if coef is None else np.ravel(coef), index=columns)


def walk_forward(ev: pd.DataFrame, X: pd.DataFrame, horizon: int, kind: str = "lasso",
                 refit_every_days: int = 63, min_train: int = 400, cv_folds: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample predictions and the coefficient path.

    A model refitted on date d is trained on events whose outcome was observed
    before d (``end_H < d``) and predicts events with ``t0 >= d`` until the next refit.
    """
    y, t0, end = ev[f"ab_{horizon}"], ev["t0"], ev[f"end_{horizon}"]
    pred = pd.Series(np.nan, index=ev.index)
    paths = {}
    d, last = t0.min(), t0.max()
    step = pd.Timedelta(days=refit_every_days)
    while d <= last:
        train = (end < d) & y.notna()
        test = (t0 >= d) & (t0 < d + step)
        if train.sum() >= min_train and test.any():
            order = t0[train].sort_values().index      # time order matters for the inner CV
            cols = X.columns[X.loc[order].std() > 0]
            model = make_model(kind, cv_folds).fit(X.loc[order, cols], y.loc[order])
            pred[test] = model.predict(X.loc[test, cols])
            paths[d] = coefficients(model, cols).reindex(X.columns).fillna(0.0)
        d += step
    return pred, pd.DataFrame(paths).T


def evaluate(pred: pd.Series, y: pd.Series, when: pd.Series | None = None, quantile: float = 0.2,
             min_per_month: int = 20) -> dict:
    """Out-of-sample quality of a score.

    Pooled figures mix two things: whether the score ranks the purchases of a
    given period, and whether its average level moves with time. The second
    is not selection skill. With ``when`` (the date of each event) the ranking
    is therefore also judged month by month: rank correlation and top-minus-rest
    spread inside each month, averaged over months, with a t-statistic across
    months. A month in which the score is the same for every purchase (a sparse
    model that selected nothing) has no ranking and is counted as such.
    """
    ok = pred.notna() & y.notna()
    p, r = pred[ok], y[ok]
    if len(p) < 30:
        return {"n": int(len(p))}
    ic = spearmanr(p, r).statistic if p.nunique() > 1 else np.nan
    top = r[p >= p.quantile(1 - quantile)]
    rest = r[p < p.quantile(1 - quantile)]
    out = {"n": int(len(p)), "rank_ic": float(ic) if np.isfinite(ic) else 0.0,
           "top_mean": float(top.mean()), "all_mean": float(r.mean()),
           "top_minus_rest": float(top.mean() - rest.mean()) if len(rest) else 0.0,
           "top_tstat": float(top.mean() / (top.std(ddof=1) / np.sqrt(len(top)))) if len(top) > 2 else np.nan}
    if when is not None:
        ics, spreads, months, flat = [], [], 0, 0
        for _, idx in p.groupby(when.reindex(p.index).dt.to_period("M")).groups.items():
            if len(idx) < min_per_month:
                continue
            months += 1
            pm, rm = p.loc[idx], r.loc[idx]
            if pm.nunique() < 2:
                flat += 1
                continue
            ics.append(spearmanr(pm, rm).statistic)
            cut = pm.quantile(1 - quantile)
            if (pm >= cut).any() and (pm < cut).any():
                spreads.append(rm[pm >= cut].mean() - rm[pm < cut].mean())
        ics, spreads = np.array(ics, dtype=float), np.array(spreads, dtype=float)
        ics = ics[np.isfinite(ics)]

        def tstat(x):
            return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan

        out.update({"months": months, "months_without_ranking": flat,
                    "monthly_ic_mean": float(ics.mean()) if len(ics) else np.nan, "monthly_ic_t": tstat(ics),
                    "monthly_spread_mean": float(spreads.mean()) if len(spreads) else np.nan,
                    "monthly_spread_t": tstat(spreads)})
    return out


def past_winner_score(ev: pd.DataFrame, horizon: int, min_trades: int = 5) -> pd.Series:
    """Naive rule used by retail trackers: follow members whose past disclosed purchases did well.

    For each event, the member's average abnormal return over his earlier
    purchases whose outcome was already known at t0.
    """
    out = pd.Series(np.nan, index=ev.index)
    for _, g in ev.groupby("member_key"):
        g = g.sort_values("t0")
        y, end, t0 = g[f"ab_{horizon}"].values, g[f"end_{horizon}"].values, g["t0"].values
        for k in range(len(g)):
            known = (end[:k] < t0[k]) & ~np.isnan(y[:k])
            if known.sum() >= min_trades:
                out[g.index[k]] = y[:k][known].mean()
    return out
