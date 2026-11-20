"""The research run: does any rule built on disclosed purchases beat a fair benchmark?

    python scripts/run_research.py               # development period only (the last months stay locked)
    python scripts/run_research.py --final       # once, when every choice is frozen: opens the locked period
    python scripts/run_research.py --synthetic   # simulated market with a planted edge (pipeline check)

Writes results/research.json and results/equity_curves.png. Every number is
out of sample: models are refitted walk-forward and only ever see outcomes
that were fully observed before the date they predict.
"""
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csig.config import ROOT, load_config
from csig.data.trades import signals, stock_trades
from csig.features.events import FEATURES, build_events
from csig.models import rpca, sparse
from csig.portfolio import backtest, metrics

warnings.filterwarnings("ignore")


def load_real(cfg):
    """Reads the committed research snapshot (data/store), so a run is reproducible without any download."""
    from csig.data.legislators import Legislators, ensure_repo
    from csig.data.prices import clean_prices
    cache, store = ROOT / cfg["data"]["cache_dir"], ROOT / cfg["data"]["store_dir"]
    trades = pd.read_parquet(store / "trades.parquet")
    st = signals(trades, cfg)
    prices, cleaning = clean_prices(pd.read_parquet(store / "panel" / "prices_adj.parquet").astype("float64"))
    prices = prices.loc[:, prices.notna().sum() > 60]
    elig = pd.read_parquet(store / "eligibility.parquet")
    sic = json.loads((store / "sic.json").read_text())
    state = json.loads((store / "state.json").read_text()) if (store / "state.json").exists() else {}
    leg = Legislators(ensure_repo(cache / "congress-legislators"))
    years = pd.date_range(st["disclosure_date"].min(), st["disclosure_date"].max(), freq="QS")
    committees = leg.committee_snapshots(years)
    fpath = store / "panel" / "factors.parquet"
    factors = pd.read_parquet(fpath) if fpath.exists() else None
    missing = sorted(set(st["ticker"]) - set(prices.columns))
    dropped = {"tickers_without_prices": len(missing),
               "share_of_trades_without_prices": float(st["ticker"].isin(missing).mean()),
               "prices_to": str(prices.index.max().date()),
               "tickers_dropped_for_broken_prices": len(cleaning["tickers_dropped_for_broken_prices"]),
               "tickers_with_isolated_fix": len(cleaning["tickers_with_isolated_fix"])}
    return st, prices, elig, leg, committees, sic, state, factors, dropped


def load_synthetic(cfg):
    from csig.sim import synthetic_market
    trades, prices, informed = synthetic_market(seed=11, edge=0.12, trades_per_member=80)
    print("synthetic market, informed members:", informed)
    return stock_trades(trades), prices, None, None, None, None, None, None, {}


def main(synthetic: bool, n_null: int, final: bool) -> None:
    cfg = load_config()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    bench, H = cfg["data"]["benchmark"], cfg["labels"]["main_horizon"]
    st, prices, elig, leg, committees, sic, state, factors, dropped = (load_synthetic if synthetic else load_real)(cfg)

    # ---- locked period: invisible until --final ---------------------------------
    fixed = cfg["research"].get("locked_from")           # fixed once the data started to be refreshed every day
    lock_start = pd.Timestamp(str(fixed)) if fixed and not synthetic else prices.index.max() - pd.DateOffset(months=cfg["research"]["lockbox_months"])
    if not final and not synthetic:
        prices = prices[prices.index < lock_start]
        st = st[st["disclosure_date"] < lock_start]

    mv_path = ROOT / cfg["data"]["store_dir"] / "market_value.parquet"
    market_value = pd.read_parquet(mv_path).astype("float64") if (not synthetic and mv_path.exists()) else None
    eval_start = pd.Timestamp(cfg["research"].get("eval_start", "1900-01-01")) if not synthetic else pd.Timestamp("1900-01-01")
    buys = st[st["side"].isin(cfg["events"]["sides"])]
    ev = build_events(buys, prices, bench, cfg["labels"]["horizons"], leg, committees, sic,
                      cfg["labels"]["winsor"], state)
    ev["unusual"] = rpca.rolling_unusual(ev, buys, cfg["rpca"]["window_days"], cfg["rpca"]["lam_scale"]).fillna(0.5)
    y = ev[f"ab_{H}"]
    report = {"synthetic": synthetic, "final_run": final, "locked_from": str(lock_start.date()),
              "events": int(len(ev)), "members": int(ev["member_key"].nunique()),
              "events_by_chamber": ev["chamber"].value_counts().to_dict() if "chamber" in ev else {},
              "from": str(ev["t0"].min().date()), "to": str(ev["t0"].max().date()),
              "outcome_sd": float(y.std()), "outcome_mean_all_purchases": float(y.mean()),
              "horizons_mean": {str(h): float(ev[f"ab_{h}"].mean()) for h in cfg["labels"]["horizons"]},
              "data_gaps": dropped,
              "feature_means": ev[FEATURES].mean().round(4).to_dict()}

    # ---- disclosed sales: informative or not? ---------------------------------
    sells = st[st["side"] == "sell"]
    vetoes = None
    if len(sells):
        sev = build_events(sells, prices, bench, [H], winsor=cfg["labels"]["winsor"])
        small = sev["filing_size"] <= cfg["portfolio"]["veto_max_filing_size"]
        report["sales"] = {"n": int(len(sev)), "outcome_mean_after_sale": float(sev[f"ab_{H}"].mean()),
                           "outcome_mean_small_filings": float(sev.loc[small, f"ab_{H}"].mean()),
                           "outcome_mean_large_filings": float(sev.loc[~small, f"ab_{H}"].mean())}
        vetoes = sev.loc[small, ["t0", "ticker"]]

    # ---- 1. prediction quality, out of sample --------------------------------
    m = cfg["model"]
    scores = {"most_bought": pd.Series(1.0, index=ev.index),      # the stocks with the most disclosed purchases
              "past_winners": sparse.past_winner_score(ev, H)}
    designs = {"": sparse.design_matrix(ev, m["min_member_trades"]),
               "+unusual": sparse.design_matrix(ev, m["min_member_trades"], FEATURES + ["unusual"])}
    report["models"], paths = {}, {}
    judged = ev["t0"] >= eval_start                    # rules are judged on this period only
    for kind, tag in [("lasso", ""), ("lasso", "+unusual"), ("omp", ""), ("ridge", ""), ("ols", ""), ("gbm", "")]:
        name = kind + tag
        pred, path = sparse.walk_forward(ev, designs[tag], H, kind, m["refit_every_days"], m["min_train_events"], m["cv_folds"])
        scores[name], paths[name] = pred, path
        report["models"][name] = sparse.evaluate(pred[judged], y[judged], ev["t0"])
        if len(path):
            report["models"][name]["nonzero_coefficients_median"] = float((path != 0).sum(axis=1).median())
    report["models"]["past_winners"] = sparse.evaluate(scores["past_winners"][judged], y[judged], ev["t0"])
    if len(paths.get("lasso", [])):
        last = paths["lasso"].iloc[-1]
        report["lasso_selected_last_fit"] = last[last != 0].sort_values(ascending=False).round(5).to_dict()
        report["lasso_nonzero_by_refit"] = {str(pd.Timestamp(d).date()): int(n) for d, n in (paths["lasso"] != 0).sum(axis=1).items()}
        chosen = (paths["lasso"] != 0).mean()
        report["lasso_selection_frequency"] = chosen[chosen > 0].sort_values(ascending=False).head(30).round(2).to_dict()
    # stability over time: the average purchase, year by year (a rule that works only in a chosen window is not a rule)
    by_year = ev[judged].groupby(ev.loc[judged, "t0"].dt.year)[f"ab_{H}"]
    report["by_year"] = {str(k): {"purchases": int(g.notna().sum()), "mean": float(g.mean()),
                                  "t": float(g.mean() / (g.std(ddof=1) / np.sqrt(g.notna().sum()))),
                                  "share_beating_benchmark": float((g.dropna() > 0).mean())} for k, g in by_year}
    report["models"]["rpca_unusual"] = sparse.evaluate(ev["unusual"][judged], y[judged], ev["t0"])   # the Robust PCA score used alone
    hi, lo = ev["unusual"] >= ev["unusual"].quantile(0.8), ev["unusual"] <= ev["unusual"].quantile(0.2)
    report["rpca_unusual_vs_herd"] = {"unusual_top20_mean": float(y[hi].mean()), "herd_bottom20_mean": float(y[lo].mean()),
                                      "difference": float(y[hi].mean() - y[lo].mean())}

    # ---- 2. portfolios --------------------------------------------------------
    returns = prices.pct_change(fill_method=None).iloc[1:]
    bench_ret, stock_ret = returns[bench], returns.drop(columns=[bench])
    pcfg = cfg["portfolio"]
    report["portfolios"], curves = {}, {}
    start = ev.loc[scores["lasso"].notna(), "t0"].min() if scores["lasso"].notna().any() else ev["t0"].min()
    start = max(start, eval_start)
    report["judged_from"] = str(pd.Timestamp(start).date())

    def record(label, res, with_null, e):
        d = res["daily"]["ret"]
        if len(d) < 60:
            return
        row = metrics.summary(d, bench_ret)
        row.update(metrics.alpha_interval(d, bench_ret))
        if factors is not None:
            row.update(metrics.factor_alpha(d, factors))
        row["avg_names"], row["avg_exposure"] = float(res["daily"]["n_names"].mean()), float(res["daily"]["exposure"].mean())
        row["annual_turnover"] = float(res["daily"]["turnover"].sum() / (len(d) / 252))
        if with_null and n_null:
            null = backtest.random_portfolios(res, stock_ret, pcfg, e, n_sims=n_null, seed=7)
            row["vs_random_eligible"] = metrics.null_pvalue(d, null)
            if market_value is not None:              # the fairer benchmark: stand-ins of similar size
                null = backtest.random_portfolios(res, stock_ret, pcfg, e, n_sims=n_null, seed=7, size=market_value)
                row["vs_random_same_size"] = metrics.null_pvalue(d, null)
        if final and not synthetic:
            locked = d[d.index >= lock_start]
            if len(locked) >= 60:
                row["locked_period"] = metrics.summary(locked, bench_ret)
        report["portfolios"][label] = row
        curves[label] = (1 + d).cumprod()

    main_e = elig if elig is not None else None
    main_tag = "screened" if elig is not None else "all_stocks"
    for name, sc in scores.items():
        sig = ev.assign(score=sc)[["t0", "ticker", "score"]]
        sig = sig[(sig["t0"] >= start) & (sig["score"] > 0)]       # all rules are compared over the same period
        record(f"{name}|{main_tag}", backtest.run_backtest(sig, stock_ret, pcfg, main_e, "equal", H), True, main_e)
        if elig is not None and name in ("most_bought", "lasso"):       # what the screen costs or brings
            record(f"{name}|unscreened", backtest.run_backtest(sig, stock_ret, pcfg, None, "equal", H), False, None)
        if vetoes is not None and name in ("most_bought", "lasso"):     # do disclosed sales help as an exit?
            record(f"{name}|{main_tag}+sale_veto",
                   backtest.run_backtest(sig, stock_ret, pcfg, main_e, "equal", H, vetoes), False, main_e)
    # the plainest rule of all: hold every stock with a recent disclosed purchase, in equal parts
    wide = {**pcfg, "max_names": 10 ** 6, "max_weight": 1.0, "min_weight": 0.0}
    sig = ev.assign(score=1.0)[["t0", "ticker", "score"]]
    sig = sig[sig["t0"] >= start]
    record(f"copy_everything|{main_tag}", backtest.run_backtest(sig, stock_ret, wide, main_e, "equal", H), False, main_e)
    if elig is not None:
        record("copy_everything|unscreened", backtest.run_backtest(sig, stock_ret, wide, None, "equal", H), False, None)
    if scores["lasso"].notna().any():
        sig = ev.assign(score=scores["lasso"])[["t0", "ticker", "score"]].query("score > 0")
        sig = sig[sig["t0"] >= start]
        record("lasso|optimizer", backtest.run_backtest(sig, stock_ret, pcfg, main_e, "optimizer", H), False, main_e)

    trial = [v["sharpe"] for v in report["portfolios"].values()]
    for label, d in curves.items():
        report["portfolios"][label].update(metrics.deflated_sharpe(d.pct_change().dropna(), trial))

    if curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        first = min(c.index[0] for c in curves.values())
        b = (1 + bench_ret[bench_ret.index >= first]).cumprod()
        ax.plot(b.index, b / b.iloc[0], color="black", lw=2, label=f"{bench} (benchmark)")
        for label, c in curves.items():
            ax.plot(c.index, c, lw=1.1, label=label)
        ax.set_yscale("log")
        ax.set_ylabel("Growth of 1 (log scale, after costs)")
        ax.set_title("Out-of-sample portfolios built on disclosed purchases" + (" - SYNTHETIC DATA" if synthetic else ""))
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / ("equity_curves_synthetic.png" if synthetic else "equity_curves.png"), dpi=150)
    if curves and not synthetic:                      # growth of 1 for every rule and the benchmark, for the report's figures
        table = pd.DataFrame(curves)
        b = (1 + bench_ret.reindex(table.index).fillna(0.0)).cumprod()
        table.insert(0, bench, b / b.iloc[0])
        table.astype("float32").to_parquet(out / ("growth_final.parquet" if final else "growth.parquet"))
    name = "research_synthetic.json" if synthetic else ("research_final.json" if final else "research.json")
    (out / name).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report.get(k) for k in ("events", "data_gaps", "sales", "models", "rpca_unusual_vs_herd")}, indent=1, default=float))
    for k, v in report["portfolios"].items():
        print(f"{k:34s} cagr={v['cagr']:+.3f} sharpe={v['sharpe']:.2f} alpha={v['alpha_annual']:+.3f} beta={v['beta']:.2f} "
              f"maxdd={v['max_drawdown']:.2f} ff_alpha={v.get('factor_alpha_annual', float('nan')):+.3f} "
              f"dsr={v.get('deflated_sharpe', float('nan')):.2f} "
              f"p_vs_random={v.get('vs_random_eligible', {}).get('p_value', float('nan')):.2f} "
              f"p_same_size={v.get('vs_random_same_size', {}).get('p_value', float('nan')):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--final", action="store_true", help="open the locked period (run once, when all choices are frozen)")
    ap.add_argument("--null", type=int, default=200, help="number of random-portfolio simulations")
    a = ap.parse_args()
    main(a.synthetic, a.null, a.final)
