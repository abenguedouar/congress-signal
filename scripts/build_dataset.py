"""Pull everything the research needs. Runs on GitHub Actions (open internet).

    python scripts/build_dataset.py                    # everything, incremental
    python scripts/build_dataset.py --stage ingest     # filings only (download + parse)
    python scripts/build_dataset.py --stage parse      # rebuild the trade table from stored text, no network
    python scripts/build_dataset.py --stage market     # prices, company data and the mandate screen
    python scripts/build_dataset.py --stage panel      # research snapshot only (from the cache)
    python scripts/build_dataset.py --stage screen     # recompute the mandate screen from stored inputs, no network
    python scripts/build_dataset.py --probe            # 20 filings per source: checks that the sources answer

Outputs (data/store): house_text/ (text of every electronic House filing),
trades.parquet, ingest_log.parquet, eligibility.parquet, sic.json, state.json,
prices_missing.json, coverage.json, and panel/ (the research snapshot: total-return
prices and factor returns, so that a research run is reproducible offline).
Re-downloadable material goes to data/cache.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from csig.config import ROOT, load_config
from csig.data import fundamentals as fnd
from csig.data import house, senate
from csig.data.legislators import Legislators, ensure_repo
from csig.data.factors import load_factors
from csig.data.prices import load_prices, update_prices
from csig.data.trades import finalize, signals

try:                                   # the mandate's rules are not part of the public copy of this repository
    from csig.screen import accounts
    from csig.screen.table import eligibility_table, load_inputs
except ImportError:
    accounts = None

FAILED = ("http", "error")           # statuses that are tried again on the next run


def ingest(cfg, cache, store, probe, years, skip_senate, minutes, offline=False) -> pd.DataFrame:
    limit = 20 if probe else None
    leg_dir = ensure_repo(cache / "congress-legislators", update=not offline)
    leg = Legislators(leg_dir)

    deadline = time.time() + minutes * 60
    trades = pd.read_parquet(store / "trades.parquet") if (store / "trades.parquet").exists() else pd.DataFrame()
    log_path = store / "ingest_log.parquet"
    log_all = pd.read_parquet(log_path) if log_path.exists() else pd.DataFrame(columns=["key", "source", "status", "n"])

    def log_add(new_log, source):
        nonlocal log_all
        if not len(new_log):
            return
        key = new_log["doc_id"] if "doc_id" in new_log else new_log["url"]
        add = pd.DataFrame({"key": key.astype(str), "source": source, "status": new_log["status"], "n": new_log["n"]})
        log_all = pd.concat([log_all[~log_all["key"].isin(add["key"])], add], ignore_index=True)

    # ---------------------------------------------------------------- House: text first, then parse
    texts = house.load_texts(store)
    this_year = pd.Timestamp.now().year
    if years:                                        # "2014-2016" or "2025": restrict the House years
        a, _, b = years.partition("-")
        year_list = list(range(int(b or a), int(a) - 1, -1))
    else:
        year_list = [this_year] if probe else list(range(this_year, cfg["data"]["start_year"] - 1, -1))
    for year in ([] if offline else year_list):      # most recent first
        if time.time() > deadline:
            print(f"[house {year}] not started: time budget used up, rerun to continue")
            continue
        scans = set(log_all.loc[(log_all["source"] == "house") & (log_all["status"] == "scan"), "key"])
        try:
            new, log = house.fetch_year(year, set(texts["doc_id"]) | scans, limit=limit, deadline=deadline)
        except Exception as exc:
            print(f"[house {year}] FAILED: {type(exc).__name__}: {exc}")
            continue
        log_add(log, "house")
        if len(new):
            texts = pd.concat([texts, new], ignore_index=True).drop_duplicates("doc_id", keep="last")
            if not probe:
                house.save_texts(texts, store, set(pd.to_datetime(new["filing_date"]).dt.strftime("%Y-%m")))
        print(f"[house {year}] new text {len(new)}  without text {log['status'].value_counts().to_dict() if len(log) else {}}")

    house_trades, parse_log = house.parse_store(texts, leg)
    log_add(parse_log, "house")
    is_house = trades["chamber"].eq("house") if len(trades) else pd.Series(dtype=bool)
    if len(trades):                                  # rows of filings whose text is not stored yet are kept as they are
        doc = trades["trade_id"].str.split("-").str[1]
        legacy = trades[is_house & ~doc.isin(set(texts["doc_id"]))]
        trades = pd.concat([trades[~is_house], legacy, house_trades], ignore_index=True)
    else:
        trades = house_trades
    if len(house_trades):
        print(f"[house] filings with text {len(texts)}  trades {len(house_trades)}  "
              f"asset types {house_trades['asset_type'].value_counts().to_dict()}  "
              f"identified by {house_trades['id_src'].value_counts().to_dict()}")
        if probe:
            print(house_trades[["member_name", "member_id", "owner", "ticker", "asset_type", "side", "trade_date",
                                "disclosure_date", "amount_min", "amount_max"]].tail(30).to_string())

    def save():
        if not probe and len(trades):
            finalize(trades).to_parquet(store / "trades.parquet")
            log_all.to_parquet(log_path)

    save()

    # ---------------------------------------------------------------- Senate
    try:
        if skip_senate or offline:
            raise senate.SenateBlocked("skipped")
        if time.time() > deadline:
            raise senate.SenateBlocked("time budget used up, rerun to continue")
        done = set(log_all.loc[(log_all["source"] == "senate") & ~log_all["status"].str.startswith(FAILED), "key"])
        since = "01/01/%d" % (this_year if probe else cfg["data"]["start_year"])
        t, log = senate.ingest(since, leg, done, limit=limit, deadline=deadline)
        log_add(log, "senate")
        if len(t):
            trades = pd.concat([trades, t], ignore_index=True)
        print(f"[senate] filings {len(log)}  trades {len(t)}  status {log['status'].value_counts().to_dict() if len(log) else {}}")
        if probe and len(t):
            print(t[["member_name", "member_id", "owner", "ticker", "side", "trade_date", "disclosure_date", "amount_min"]].head(12).to_string())
    except senate.SenateBlocked as exc:
        print(f"[senate] not read ({exc})")
    except Exception as exc:
        print(f"[senate] FAILED: {type(exc).__name__}: {exc}")
    save()

    if not len(trades):
        raise SystemExit("no trades ingested")
    trades = finalize(trades)
    print(f"[store] trades {len(trades)}  by chamber {trades['chamber'].value_counts().to_dict()}  "
          f"filings logged {log_all.groupby('source')['status'].value_counts().to_dict()}")
    return trades


def market(cfg, cache, store, trades, probe) -> None:
    st = signals(trades, cfg)
    tickers = sorted(st["ticker"].unique())
    if probe:
        tickers = tickers[:25]
    asof = st.groupby("ticker")["trade_date"].median().dt.strftime("%Y-%m-%d").to_dict()   # a day the symbol was in use
    status = update_prices(tickers + [cfg["data"]["benchmark"]], cache, asof=asof)
    priced = [t for t in tickers if status.get(t) != "missing"]
    print(f"[prices] {pd.Series(status).value_counts().to_dict()}")
    meta = json.loads((cache / "prices" / "_meta.json").read_text())
    print(f"[prices] by source {pd.Series({t: meta.get(t, {}).get('src', '?') for t in tickers}).value_counts().to_dict()}")
    n_trades = st["ticker"].value_counts()
    missing = {t: int(n_trades[t]) for t in tickers if t not in priced}
    (store / "prices_missing.json").write_text(json.dumps(dict(sorted(missing.items(), key=lambda kv: -kv[1])), indent=0))
    print(f"[prices] tickers without prices {len(missing)}, carrying {sum(missing.values())} of {len(st)} stock trades")

    try:
        tmap = fnd.ticker_map(cache)
    except Exception as exc:
        raise SystemExit(f"[sec] ticker list refused ({type(exc).__name__}: {exc}); check the SEC_USER_AGENT secret")

    keep, version = (accounts.keep_tags(), accounts.FACTS_VERSION) if accounts else (None, 0)

    def one(t):
        try:
            co = fnd.company(tmap[t], cache, keep, version)
            return t, {k: co.get(k, "") for k in ("cik", "name", "sic", "sic_description", "state")}, None
        except Exception as exc:
            return t, None, type(exc).__name__

    companies, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:   # stays under the SEC limit of 10 requests per second
        for t, info, err in pool.map(one, [t for t in priced if t in tmap]):
            if info is None:
                errors[t] = err
            else:
                companies[t] = info
    sic = {t: c["sic"] for t, c in companies.items()}
    state = {t: c["state"] for t, c in companies.items()}
    print(f"[sec] companies {len(sic)}  errors {len(errors)} {dict(list(errors.items())[:10])}")
    (store / "sic.json").write_text(json.dumps(sic))
    (store / "state.json").write_text(json.dumps(state))
    panel = (cache / "probe_panel") if probe else (store / "panel")
    coverage = {
        "run": str(pd.Timestamp.now()), "probe": probe,
        "trades": int(len(trades)), "stock_trades": int(len(st)),
        "by_chamber_year": {f"{c}-{y}": int(n) for (c, y), n in
                            st.groupby(["chamber", st["disclosure_date"].dt.year]).size().items()},
        "member_id_resolved": float(st["member_id"].notna().mean()),
        "tickers": len(tickers), "tickers_priced": len(priced), "tickers_with_sec_data": len(sic),
        "stock_trades_priced": int(st["ticker"].isin(priced).sum()),
    }
    if accounts is not None:
        export_screen_inputs(cache, panel, companies, priced)
        coverage.update(screen(cfg, store, st, panel, probe))
    else:
        print("[screen] the mandate's rules are not in this copy: data/store/eligibility.parquet is used as provided")
    (store / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print(json.dumps(coverage, indent=1))


def screen(cfg, store, st, panel, probe=False) -> dict:
    """The mandate screen on every quarter end and on the latest day, from the files of ``panel`` (no download)."""
    inputs = load_inputs(panel)
    tickers = sorted(set(st["ticker"]) & set(inputs["close"].columns))
    quarters = pd.date_range(st["disclosure_date"].min().normalize(), pd.Timestamp.now().normalize(), freq="QE")
    quarters = quarters[-2:] if probe else quarters
    days = inputs["close"].dropna(how="all").index
    last = days[days <= pd.Timestamp.now().normalize()].max()      # latest day with prices
    quarters = [q for q in quarters if q < last]
    elig, audit, caps = eligibility_table(inputs, tickers, quarters + [last], cfg["screen"])
    if not probe:
        elig.to_parquet(store / "eligibility.parquet")
        audit.to_parquet(store / "screen_audit.parquet")
        caps.dropna(axis=1, how="all").to_parquet(store / "market_value.parquet", compression="zstd")
    out = {"screen_date": str(pd.Timestamp(last).date()), "screened_tickers": len(tickers),
           "with_usable_accounts": int(audit["balance_date"].notna().sum()) if "balance_date" in audit else 0,
           "eligible_today": int(elig.iloc[-1].sum()), "eligible_share_today": float(elig.iloc[-1].mean()),
           "eligible_share_by_year": {str(y): round(float(v), 3) for y, v in elig.groupby(elig.index.year).mean().mean(axis=1).items()}}
    if len(audit):
        why = audit.loc[~audit["eligible"], "reasons"].str.split("; ").explode().str.replace(r"^[A-D]: ", "", regex=True)
        out["refusal_reasons"] = why.value_counts().head(12).to_dict()
    return out



def export_screen_inputs(cache, panel, companies, priced) -> None:
    """Everything the mandate screen reads, in compact form, so that it can be recomputed and audited offline:
    company facts as filed (with filing dates), month-end closes and stock splits."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    panel.mkdir(parents=True, exist_ok=True)
    (panel / "companies.json").write_text(json.dumps(companies))
    writer, n_rows = None, 0
    for t, info in companies.items():
        table = accounts.facts_table({t: fnd.company(info["cik"], cache, accounts.keep_tags(), accounts.FACTS_VERSION)})
        if table.empty:
            continue
        # the same figure is repeated in later filings: the first filing date is the one that matters
        table = (table.sort_values("filed").drop_duplicates(["taxo", "tag", "unit", "start", "end", "val"], keep="first"))
        part = pa.Table.from_pandas(table.astype({c: "string" for c in ("ticker", "taxo", "tag", "unit", "start", "end", "filed")}),
                                    preserve_index=False)
        writer = writer or pq.ParquetWriter(panel / "sec_facts.parquet", part.schema, compression="zstd")
        writer.write_table(part)
        n_rows += len(table)
    if writer:
        writer.close()
    close = load_prices(priced, cache, "close")
    monthly = close.resample("ME").last()
    monthly = pd.concat([monthly[monthly.index <= close.index.max()], close.iloc[[-1]]])   # full months, then the latest day
    monthly = monthly[~monthly.index.duplicated(keep="last")].sort_index().astype("float32")
    monthly.to_parquet(panel / "close_monthly.parquet", compression="zstd")
    splits = []
    for t in priced:
        f = cache / "prices" / f"{t}.parquet"
        sp = pd.read_parquet(f, columns=["split"])["split"]
        sp = sp[sp > 0]
        splits += [(t, d, float(v)) for d, v in sp.items()]
    pd.DataFrame(splits, columns=["ticker", "date", "ratio"]).to_parquet(panel / "splits.parquet")
    print(f"[panel] screen inputs: {n_rows} facts of {len(companies)} companies "
          f"({(panel / 'sec_facts.parquet').stat().st_size / 1e6:.1f} MB), {monthly.shape[0]} month-ends, {len(splits)} splits")


def export_panel(cfg, cache, store, trades) -> None:
    """Research snapshot: what scripts/run_research.py reads, frozen in the repository."""
    st = signals(trades, cfg)
    panel = store / "panel"
    panel.mkdir(exist_ok=True)
    adj = load_prices(sorted(st["ticker"].unique()) + [cfg["data"]["benchmark"]], cache, "adj")
    adj = adj[adj.index >= "2013-01-01"].astype("float32")
    adj.to_parquet(panel / "prices_adj.parquet", compression="zstd")
    print(f"[panel] prices {adj.shape[0]} days x {adj.shape[1]} tickers, to {adj.index.max().date()}, "
          f"{(panel / 'prices_adj.parquet').stat().st_size / 1e6:.1f} MB")
    try:
        f = load_factors(cache, refresh=True)
        f[f.index >= "2013-01-01"].to_parquet(panel / "factors.parquet")
        print(f"[panel] factors to {f.index.max().date()}")
    except Exception as exc:
        print(f"[panel] factors unavailable: {type(exc).__name__}: {exc}")


def main(stage: str = "all", probe: bool = False, years: str = "", skip_senate: bool = False, minutes: float = 300.0) -> None:
    cfg = load_config()
    cache, store = ROOT / cfg["data"]["cache_dir"], ROOT / cfg["data"]["store_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)
    if stage in ("market", "panel", "screen"):
        trades = pd.read_parquet(store / "trades.parquet")
    else:
        trades = ingest(cfg, cache, store, probe, years, skip_senate, minutes, offline=(stage == "parse"))
    if stage in ("all", "market"):
        market(cfg, cache, store, trades, probe)
    if stage in ("all", "market", "panel") and not probe:
        export_panel(cfg, cache, store, trades)
    if stage == "screen":                                # offline: recompute the screen from the stored inputs
        print(json.dumps(screen(cfg, store, signals(trades, cfg), store / "panel"), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "ingest", "parse", "market", "panel", "screen"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--years", default="", help='e.g. "2014-2016"')
    ap.add_argument("--skip-senate", action="store_true")
    ap.add_argument("--minutes", type=float, default=300.0, help="time budget for downloading filings; rerun to continue")
    a = ap.parse_args()
    main(a.stage, a.probe, a.years, a.skip_senate, a.minutes)
