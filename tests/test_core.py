"""Tests on fixtures and on a synthetic market with a planted edge."""
import numpy as np
import pandas as pd
import pytest

from csig.config import load_config
from csig.data.house import parse_ptr_text
from csig.data.trades import clean_ticker, finalize, parse_amount, stock_trades
from csig.features.events import build_events
from csig.models import rpca, sparse
from csig.portfolio import backtest, metrics, optimizer
from csig.sim import synthetic_market

CFG = load_config()


# ---------------------------------------------------------------- parsing
def test_amounts_and_tickers():
    assert parse_amount("$1,001 - $15,000") == (1001.0, 15000.0)
    assert parse_amount("Over $50,000,000") == (50000000.0, None)
    assert parse_amount("Unknown") == (None, None)
    assert clean_ticker('<a href="x">BRK.B</a>') == "BRK.B"
    assert clean_ticker("--") is None and clean_ticker("N/A") is None


def test_house_ptr_text_both_layouts():
    old = ("ID Owner Asset Transaction Date Notification Amount\n"
           "SP Amazon.com, Inc. (AMZN) [ST] P 01/02/2019 01/05/2019 $1,001 - $15,000\n"
           "FILING STATUS: New\n"
           "JT Apple Inc. (AAPL) [ST] S (partial) 02/03/2019 02/04/2019 $15,001 - $50,000\n")
    rows = parse_ptr_text(old)
    assert [(r["ticker"], r["side"], r["owner"]) for r in rows] == [("AMZN", "buy", "spouse"), ("AAPL", "sell", "joint")]
    new = ("Microsoft Corporation - Common Stock\n(MSFT) [ST]\nP03/16/202603/18/2026$50,001 -\n$100,000\n"
           "F S: New\nS O: Trust\nUS Treasury Bill [GS]\nS03/17/202603/18/2026Over $1,000,000\n")
    rows = parse_ptr_text(new)
    assert rows[0]["ticker"] == "MSFT" and rows[0]["amount_max"] == 100000.0
    assert rows[1]["asset_type"] == "other" and rows[1]["ticker"] is None


# ---------------------------------------------------------------- robust PCA
def test_pcp_recovers_low_rank_plus_sparse():
    rng = np.random.default_rng(1)
    n, r = 60, 3
    L0 = rng.normal(size=(n, r)) @ rng.normal(size=(r, n))
    S0 = np.zeros((n, n))
    mask = rng.random((n, n)) < 0.05
    S0[mask] = rng.choice([-8, 8], size=mask.sum())
    fit = rpca.pcp(L0 + S0)
    assert fit["rank"] == r
    assert np.linalg.norm(fit["L"] - L0) / np.linalg.norm(L0) < 1e-3
    assert np.linalg.norm(fit["S"] - S0) / np.linalg.norm(S0) < 1e-3


# ---------------------------------------------------------------- models and backtest
@pytest.fixture(scope="module")
def planted():
    trades, prices, informed = synthetic_market(seed=3, edge=0.12, trades_per_member=80)
    ev = build_events(stock_trades(trades), prices, "SPY", [63], winsor=0.0)
    return trades, prices, informed, ev


def test_events_have_no_look_ahead(planted):
    _, prices, _, ev = planted
    assert (ev["t0"] > ev["disclosure_date"]).all()
    assert (ev["end_63"] > ev["t0"]).all()
    k = ev.index[10]
    p = prices[ev.at[k, "ticker"]]
    b = prices["SPY"]
    expected = (p[ev.at[k, "end_63"]] / p[ev.at[k, "t0"]] - 1) - (b[ev.at[k, "end_63"]] / b[ev.at[k, "t0"]] - 1)
    assert ev.at[k, "ab_63"] == pytest.approx(expected)


def test_lasso_recovers_planted_members(planted):
    _, _, informed, ev = planted
    X = sparse.design_matrix(ev, min_member_trades=8)
    pred, path = sparse.walk_forward(ev, X, 63, "lasso", refit_every_days=126, min_train=300)
    last = path.iloc[-1]
    members = last[[c for c in last.index if c.startswith("member::")]]
    top = set(members.nlargest(len(informed)).index.str.replace("member::", ""))
    assert top == set(informed)
    assert (members != 0).sum() <= 3 * len(informed)             # sparse: most members get exactly zero
    stats = sparse.evaluate(pred, ev["ab_63"])
    assert stats["rank_ic"] > 0.03 and stats["top_minus_rest"] > 0.02


def test_walk_forward_never_trains_on_the_future(planted):
    _, _, _, ev = planted
    X = sparse.design_matrix(ev)
    pred, _ = sparse.walk_forward(ev, X, 63, "ridge", refit_every_days=126, min_train=300)
    first = ev.loc[pred.notna(), "t0"].min()
    assert (ev["end_63"] < first).sum() >= 300                    # enough fully observed outcomes existed


def test_backtest_finds_planted_edge_and_null_is_centred(planted):
    _, prices, informed, ev = planted
    returns = prices.pct_change().iloc[1:]
    cfg = dict(CFG["portfolio"], max_names=10)
    good = ev[ev["member_key"].isin(informed)].assign(score=1.0)
    res = backtest.run_backtest(good[["t0", "ticker", "score"]], returns.drop(columns="SPY"), cfg, sizing="equal")
    s = metrics.summary(res["daily"]["ret"], returns["SPY"])
    assert s["alpha_annual"] > 0.05
    null = backtest.random_portfolios(res, returns.drop(columns="SPY"), cfg, n_sims=40, seed=1)
    assert metrics.null_pvalue(res["daily"]["ret"], null)["p_value"] < 0.05
    # followers of uninformed members must look like random portfolios
    noise = ev[~ev["member_key"].isin(informed)].assign(score=1.0)
    res0 = backtest.run_backtest(noise[["t0", "ticker", "score"]], returns.drop(columns="SPY"), cfg, sizing="equal")
    null0 = backtest.random_portfolios(res0, returns.drop(columns="SPY"), cfg, n_sims=40, seed=2)
    assert 0.05 < metrics.null_pvalue(res0["daily"]["ret"], null0)["p_value"] < 0.95


def test_backtest_accounting():
    idx = pd.bdate_range("2024-01-01", periods=30)
    returns = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    sig = pd.DataFrame({"t0": [idx[0], idx[0]], "ticker": ["A", "B"], "score": [1.0, 1.0]})
    cfg = dict(CFG["portfolio"], max_weight=0.5, cost_bps=0, signal_ttl_days=100)
    res = backtest.run_backtest(sig, returns, cfg, sizing="equal")
    assert np.allclose(res["daily"]["ret"], 0.01) and res["daily"]["exposure"].round(6).eq(1.0).all()
    cfg["cost_bps"] = 100
    res = backtest.run_backtest(sig, returns, cfg, sizing="equal")
    assert res["daily"]["ret"].iloc[0] == pytest.approx(0.01 - 1.0 * 0.01)  # full investment pays 1% once
    assert res["daily"]["ret"].iloc[1] == pytest.approx(0.01)


def test_optimizer_respects_constraints():
    rng = np.random.default_rng(0)
    names = [f"S{k}" for k in range(25)]
    mu = pd.Series(rng.uniform(0.0, 0.3, 25), index=names)
    cov = np.cov(rng.normal(0, 0.02, (300, 25)), rowvar=False) * 252
    w = optimizer.target_weights(mu, cov, None, CFG["portfolio"])
    assert (w >= 0).all() and w.sum() <= 1 + 1e-6 and w.max() <= CFG["portfolio"]["max_weight"] + 1e-6
    assert ((w == 0) | (w >= CFG["portfolio"]["min_weight"])).all() and (w > 0).sum() <= CFG["portfolio"]["max_names"]
    # a heavy turnover penalty keeps the previous book
    cfg = dict(CFG["portfolio"], turnover_penalty=10.0)
    prev = pd.Series(0.0, index=names)
    prev.iloc[:5] = 0.1
    assert np.allclose(optimizer.target_weights(mu, cov, prev, cfg), prev, atol=1e-4)


# ---------------------------------------------------------------- confirmed additions
def test_option_purchases_become_signals_on_the_stock():
    base = dict(chamber="house", member_id="X", member_name="X", owner="self", ticker="NVDA", amount_min=1001.0,
                amount_max=15000.0, trade_date="2024-01-02", disclosure_date="2024-01-20", source_url="u")
    df = pd.DataFrame([
        dict(base, trade_id="1", asset_name="NVIDIA call options, strike 100", asset_type="option", side="buy"),
        dict(base, trade_id="2", asset_name="NVIDIA put options", asset_type="option", side="buy"),
        dict(base, trade_id="3", asset_name="NVIDIA call options", asset_type="option", side="sell"),
        dict(base, trade_id="4", asset_name="NVIDIA Corporation", asset_type="stock", side="buy"),
    ])
    df["trade_date"], df["disclosure_date"] = pd.to_datetime(df["trade_date"]), pd.to_datetime(df["disclosure_date"])
    st = stock_trades(df).set_index("trade_id")
    assert list(st.index) == ["1", "2", "4"]                       # the sold call is ambiguous and dropped
    assert st.loc["1", "side"] == "buy" and st.loc["2", "side"] == "sell" and st.loc["1", "via_option"] == 1.0
    assert len(stock_trades(df, include_options=False)) == 1


def test_sale_veto_keeps_the_stock_out():
    idx = pd.bdate_range("2024-01-01", periods=40)
    returns = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    sig = pd.DataFrame({"t0": [idx[0], idx[0]], "ticker": ["A", "B"], "score": [1.0, 1.0]})
    cfg = dict(CFG["portfolio"], max_weight=0.5, cost_bps=0, signal_ttl_days=100, veto_days=1000)
    veto = pd.DataFrame({"t0": [idx[10]], "ticker": ["B"]})
    res = backtest.run_backtest(sig, returns, cfg, sizing="equal", vetoes=veto)
    books = list(res["book"].values())
    assert set(books[0].index) == {"A", "B"} and set(books[-1].index) == {"A"}


def test_factor_alpha_and_deflated_sharpe():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    f = pd.DataFrame({"MKT": rng.normal(0.0004, 0.01, 1500), "SMB": rng.normal(0, 0.005, 1500), "RF": 0.0}, index=idx)
    pure_tilt = pd.Series(1.2 * f["MKT"] + 0.5 * f["SMB"] + rng.normal(0, 0.002, 1500), index=idx)
    with_alpha = pure_tilt + 0.0006
    a0, a1 = metrics.factor_alpha(pure_tilt, f), metrics.factor_alpha(with_alpha, f)
    assert abs(a0["factor_alpha_tstat"]) < 2.5 and a1["factor_alpha_tstat"] > 5
    assert a1["loading_MKT"] == pytest.approx(1.2, abs=0.05) and a1["factor_alpha_annual"] == pytest.approx(0.151, abs=0.04)
    noise = [pd.Series(rng.normal(0, 0.01, 1500), index=idx) for _ in range(20)]
    sharpes = [metrics.summary(x)["sharpe"] for x in noise]
    best = noise[int(np.argmax(sharpes))]
    assert metrics.deflated_sharpe(best, sharpes)["deflated_sharpe"] < 0.9      # the luckiest of 20 is not convincing
    strong = pd.Series(rng.normal(0.002, 0.01, 1500), index=idx)                # Sharpe ratio above 3
    assert metrics.deflated_sharpe(strong, sharpes + [metrics.summary(strong)["sharpe"]])["deflated_sharpe"] > 0.95


def test_house_parser_on_real_filings():
    """Text extracted from two live filings during the first probe (wrapped names and amounts)."""
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures"
    rows = parse_ptr_text((fx / "house_ptr_20033718.txt").read_text())
    assert len(rows) == 4 and all(r["owner"] == "joint" and r["side"] == "buy" for r in rows)
    assert rows[0]["asset_name"] == "Floyd County IN 4.00% 12/30/2031" and rows[0]["asset_type"] == "other"
    assert (rows[0]["amount_min"], rows[0]["amount_max"]) == (100001.0, 250000.0)
    assert str(rows[0]["trade_date"].date()) == "2025-12-05"
    assert rows[1]["asset_name"] == "Fremont IN Cmnty Schs Gen Oblig Bds 4.00% 1/15/2032"
    rows = parse_ptr_text((fx / "house_ptr_20033705.txt").read_text())
    assert len(rows) == 2 and rows[0]["amount_max"] == 25000000.0
    assert rows[1]["asset_name"].startswith("Prudential RILA")


def test_house_parser_wrapped_stock_row():
    text = ("ID Owner Asset Transaction Date Notification Amount Cap.\nType Date Gains >\n$200?\n"
            "SP Taiwan Semiconductor Manufacturing S (partial) 03/02/2026 03/04/2026 $15,001 -\n"
            "Company Ltd. (TSM) [ST] $50,000\nF S : New\nD : Sold part of the position\nfor tuition.\n"
            "NVIDIA Corporation - Common Stock (NVDA) [ST] P 03/03/2026 03/04/2026 $1,001 - $15,000\nF S : New\n")
    rows = parse_ptr_text(text)
    assert [(r["ticker"], r["side"], r["owner"], r["asset_type"]) for r in rows] == [
        ("TSM", "sell", "spouse", "stock"), ("NVDA", "buy", "self", "stock")]
    assert rows[0]["amount_max"] == 50000.0 and rows[0]["asset_name"] == "Taiwan Semiconductor Manufacturing Company Ltd."


def test_house_parser_spouse_over_one_million():
    text = ("SP U.S. Treasury Note due 2/28/2029 P 12/15/2025 12/16/2025 Spouse/DC Over\n[GS] $1,000,000\nF S : New\n")
    rows = parse_ptr_text(text)
    assert len(rows) == 1 and rows[0]["amount_min"] == 1000000.0 and rows[0]["amount_max"] is None
    assert rows[0]["asset_name"] == "U.S. Treasury Note due 2/28/2029" and rows[0]["owner"] == "spouse"


def test_asset_cell_reader():
    from csig.data.assets import interpret
    a = interpret("sP apple Inc. (aaPl) [sT]")                       # broken letter case in the PDF text layer
    assert (a["owner"], a["ticker"], a["asset_type"], a["id_src"], a["asset_name"]) == ("spouse", "AAPL", "stock", "coded", "apple Inc.")
    a = interpret("2000004251 JT Corning Incorporated (gLW)")        # transaction id, owner, form without type code
    assert (a["owner"], a["ticker"], a["asset_type"], a["id_src"]) == ("joint", "GLW", "stock", "ticker")
    assert interpret("Vanguard FTSE developed Markets ETF (VEa)")["asset_type"] == "other"
    assert interpret("WELLS FARgo & Co NTS B/E 02.125% 042219")["asset_type"] == "other"
    o = interpret("TLT jaN 20 121 CaLL")
    assert (o["asset_type"], o["ticker"]) == ("option", "TLT")
    assert interpret("Family Trust > Deere & Company (DE) [ST]")["ticker"] == "DE"
    assert interpret("Berkshire Hathaway Inc. New (BRK/B) [ST]")["ticker"] == "BRK.B"
    b = interpret("Bill Barrett Corporation (BBG)")
    assert (b["owner"], b["ticker"], b["asset_type"]) == (None, "BBG", "stock")
    assert interpret("Prince georges CNTY MD go 5.00% Due 07/15/2031 [gS]")["id_src"] == "coded"


def test_house_text_store_roundtrip(tmp_path):
    from pathlib import Path
    from csig.data import house
    fx = Path(__file__).parent / "fixtures"
    texts = pd.DataFrame([
        {"doc_id": "20033718", "year": 2025, "first": "A", "last": "B", "filing_date": pd.Timestamp("2025-12-09"),
         "text": (fx / "house_ptr_20033718.txt").read_text()},
        {"doc_id": "20033705", "year": 2026, "first": "C", "last": "D", "filing_date": pd.Timestamp("2026-01-01"),
         "text": (fx / "house_ptr_20033705.txt").read_text()},
        {"doc_id": "1", "year": 2026, "first": "E", "last": "F", "filing_date": pd.Timestamp("2026-01-02"),
         "text": "a text PDF in which no transaction row is recognised " * 3},
    ])
    house.save_texts(texts, tmp_path)
    assert sorted(p.name for p in (tmp_path / "house_text").glob("*.parquet")) == ["2025-12.parquet", "2026-01.parquet"]
    back = house.load_texts(tmp_path)
    assert set(back["doc_id"]) == {"20033718", "20033705", "1"}
    trades, log = house.parse_store(back)
    assert len(trades) == 6 and set(trades["chamber"]) == {"house"} and trades["id_src"].eq("coded").all()
    assert log.set_index("doc_id")["status"].to_dict() == {"20033718": "parsed", "20033705": "parsed", "1": "no_rows"}
    assert trades["trade_id"].str.startswith("H-2003").all()


def test_house_parser_ignores_comment_tail_before_next_row():
    text = ("JT Medtronic Inc. (MDT) E 01/26/2015 02/03/2015 $15,001 - $50,000\n"
            "D : exchange between Medtronic Inc and the new\nMedtronic plc shares.\n"
            "JT Medtronic plc. ordinary Shares (MdT) E 01/26/2015 02/03/2015 $15,001 - $50,000\n")
    rows = parse_ptr_text(text)
    assert [(r["owner"], r["ticker"], r["side"], r["id_src"]) for r in rows] == [("joint", "MDT", "exchange", "ticker")] * 2
    assert rows[1]["asset_name"] == "Medtronic plc. ordinary Shares"


def test_descriptions_drive_option_signals_and_mechanical_rows():
    text = ("Walt Disney Company (DIS) [OP] P 12/17/2021 12/22/2021 $100,001 - $250,000\n"
            "F S : New\nD : Purchased 100 call options with a strike price of $100 and an\nexpiration date of 1/21/22.\n"
            "JT Coca-Cola Company (KO) [ST] P 12/15/2021 12/22/2021 $1,001 - $15,000\n"
            "F S : New\nS O : Morgan Stanley - Select UMA Account # 1\nD : Dividend reinvestment\n"
            "SP Tesla, Inc. (TSLA) [OP] S 12/16/2021 12/22/2021 $15,001 - $50,000\nF S : New\nD : Sold 5 call options.\n"
            "Apple Inc. (AAPL) [ST] P 12/16/2021 12/22/2021 $15,001 - $50,000\n"
            "F S : New\nS O : Merrill Lynch- Advisor Discretion Account\n"
            "* For the complete list of asset type abbreviations, please visit the site.\n")
    rows = parse_ptr_text(text)
    assert [r["ticker"] for r in rows] == ["DIS", "KO", "TSLA", "AAPL"]
    assert rows[0]["note"].endswith("expiration date of 1/21/22.") and rows[1]["account"].startswith("Morgan Stanley")
    assert rows[3]["note"] == "" and "Discretion" in rows[3]["account"]
    df = pd.DataFrame(rows).assign(trade_id=["a", "b", "c", "d"], chamber="house", member_id="X", member_name="X",
                                   disclosure_date=pd.Timestamp("2021-12-22"), source_url="u")
    st = stock_trades(finalize(df))
    assert sorted(st["ticker"]) == ["AAPL", "DIS"]                      # reinvested dividend and written call dropped
    dis = st.set_index("ticker").loc["DIS"]
    assert dis["side"] == "buy" and dis["via_option"] == 1.0
    assert st.set_index("ticker").loc["AAPL", "managed"] == 1.0


def test_clean_prices_drops_broken_series_and_fixes_isolated_jumps():
    from csig.data.prices import clean_prices
    idx = pd.bdate_range("2024-01-01", periods=8)
    good = pd.Series([10, 10.1, 10.2, 10.1, 10.3, 10.4, 10.5, 10.6], index=idx, dtype=float)
    one_jump = pd.Series([10, 10.1, 2000, 2020, 2040, 2060, 2080, 2100], index=idx, dtype=float)   # one absurd day
    broken = pd.Series([10, -3, 400, 2, 900, 1, 700, 5], index=idx, dtype=float)
    clean, rep = clean_prices(pd.DataFrame({"G": good, "J": one_jump, "B": broken}))
    assert list(clean.columns) == ["G", "J"] and rep["tickers_dropped_for_broken_prices"] == ["B"]
    assert rep["tickers_with_isolated_fix"] == ["J"]
    r = clean["J"].pct_change().dropna()
    assert r.abs().max() < 0.02 and abs(r.iloc[1]) < 1e-12 and abs(r.iloc[2] - 0.01) < 1e-9
    pd.testing.assert_series_equal(clean["G"], good, check_names=False)


def test_random_benchmark_draws_stand_ins_of_similar_size():
    days = pd.bdate_range("2024-01-01", periods=40)
    big, small = [f"B{i}" for i in range(6)], [f"S{i}" for i in range(12)]
    returns = pd.DataFrame(0.0, index=days, columns=big + small)
    returns[big] = 0.001                      # large companies earn 0.1% a day, small ones 1%
    returns[small] = 0.01
    cfg = {"cost_bps": 0.0}
    strategy = {"book": {days[0]: pd.Series({"B0": 0.5, "B1": 0.5})},
                "daily": pd.DataFrame({"ret": 0.001}, index=days[1:])}
    size = pd.DataFrame([[1e11] * 6 + [1e9] * 12], index=[days[0] - pd.Timedelta(days=30)], columns=big + small)
    same = backtest.random_portfolios(strategy, returns, cfg, n_sims=30, seed=3, size=size)
    anything = backtest.random_portfolios(strategy, returns, cfg, n_sims=30, seed=3)
    assert np.allclose(same.to_numpy(), 0.001)                      # stand-ins are large companies only
    assert anything.to_numpy().mean() > 0.004                       # without matching, mostly small ones
