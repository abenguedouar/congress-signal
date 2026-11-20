"""The snapshot of the test account that the page shows: built from canned answers of the broker (no network)."""
from datetime import datetime, timezone

from csig.live import snapshot

LAUNCH = "2026-11-16"


def fake(clock_time="2026-11-17T11:00:00-05:00", is_open=True, equity="101000", latest=721.0, fail=()):
    def get(url, params, key, secret):
        path = url.split(".markets", 1)[1]
        if any(f in path for f in fail):
            raise RuntimeError(path)
        if path == "/v2/account":
            return {"equity": equity, "cash": "1000", "last_equity": "100500"}
        if path == "/v2/clock":
            return {"timestamp": clock_time, "is_open": is_open, "next_open": "2026-11-18T09:30:00-05:00", "next_close": "2026-11-17T16:00:00-05:00"}
        if path == "/v2/positions":
            return [{"symbol": "AAA", "qty": "10", "market_value": "60000", "current_price": "6000", "avg_entry_price": "5900", "unrealized_pl": "1000", "change_today": "0.01"},
                    {"symbol": "BBB", "qty": "5", "market_value": "40000", "current_price": "8000", "avg_entry_price": "8100", "unrealized_pl": "-500", "change_today": "-0.02"}]
        if path == "/v2/account/portfolio/history":
            day = lambda d: int(datetime(2026, 11, d, 5, tzinfo=timezone.utc).timestamp())
            return {"timestamp": [day(12), day(13), day(16), day(17)], "equity": [100000, 100000, 100500, 100900], "base_value": 100000}
        if path == "/v2/stocks/bars" and params.get("timeframe") == "1Min":       # the minute of the first purchases
            assert params["start"] == "2026-11-16T15:00:00Z" and params["adjustment"] == "raw"
            return {"bars": {"SPY": [{"t": "2026-11-16T15:00:00Z", "c": 703.6, "vw": 703.5}]}}
        if path == "/v2/stocks/bars":
            return {"bars": {"SPY": [{"t": "2026-11-12T05:00:00Z", "c": 690.0}, {"t": "2026-11-13T05:00:00Z", "c": 700.0}, {"t": "2026-11-16T05:00:00Z", "c": 707.0}]}}
        if path == "/v2/stocks/SPY/snapshot":
            return {"latestTrade": {"p": latest}, "dailyBar": {"t": clock_time[:10] + "T05:00:00Z", "c": 714.0}}
        if path == "/v2/orders":
            return [{"id": "o2", "symbol": "BBB", "side": "buy", "filled_qty": "5", "filled_avg_price": "8100", "filled_at": "2026-11-16T15:01:00Z", "submitted_at": "2026-11-16T15:00:00Z"},
                    {"id": "o1", "symbol": "AAA", "side": "buy", "filled_qty": "10", "filled_avg_price": "5900", "filled_at": "2026-11-16T15:00:30Z", "submitted_at": "2026-11-16T15:00:00Z"},
                    {"id": "o0", "symbol": "CCC", "side": "buy", "filled_qty": "0", "filled_avg_price": None, "filled_at": None, "submitted_at": "2026-11-16T15:00:00Z"}]
        raise AssertionError(path)
    return get


CONTEXT = {"base_equity": 100000.0, "names": {"AAA": "Alpha Corp."}, "reasons": {"o1": "new in target"},
           "who": {"AAA": [{"member": "A. Member", "chamber": "house", "disclosed": "2026-11-10", "traded": "2026-10-30", "lo": 1001.0, "hi": 15000.0, "owner": "self"}]}}


def test_before_the_launch_nothing_is_shown():
    out = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake(clock_time="2026-11-13T12:00:00-05:00"))
    assert out["mode"] == "waiting" and out["positions"] == [] and out["series"]["dates"] == [] and out["equity"] == 100000.0


K = 703.5 / 700.0             # the benchmark at the minute of the first purchases, against its close before the launch


def test_series_starts_at_one_at_the_first_purchases():
    out = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake())
    S = out["series"]
    assert out["mode"] == "live" and out["market"]["open"]
    assert out["baseline"] == {"date": "2026-11-16", "at": "2026-11-16T15:00:30Z", "benchmark": 703.5, "k": round(K, 8)}
    assert S["dates"] == ["2026-11-16", "2026-11-16", "2026-11-17"]   # the purchases, the close of that day, today
    assert S["p"] == [1.0, 1.005, 1.01]                               # cash until the purchases: the account's value over 100,000
    assert S["s"] == [1.0, round(1.01 / K, 6), round(1.03 / K, 6)]    # the benchmark counted from the same minute


def test_before_any_purchase_the_series_starts_at_the_close_before_the_launch():
    base = fake(clock_time="2026-11-16T09:45:00-05:00")
    get = lambda url, params, key, secret: [] if url.endswith("/v2/orders") else base(url, params, key, secret)
    out = snapshot.build("k", "s", LAUNCH, CONTEXT, get=get)
    assert out["baseline"] is None and out["series"]["dates"][0] == "2026-11-13" and out["series"]["s"][0] == 1.0
    assert abs(out["day"] - (101000 / 100500 - 1)) < 1e-12


def test_holdings_orders_and_who():
    out = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake())
    a, b = out["positions"]
    assert (a["ticker"], a["name"], a["qty"], a["value"]) == ("AAA", "Alpha Corp.", 10.0, 60000.0) and abs(a["weight"] - 60000 / 101000) < 1e-12
    assert a["who"][0]["member"] == "A. Member" and b["name"] == "BBB" and b["who"] == []
    assert [o["ticker"] for o in out["activity"]] == ["BBB", "AAA"]             # the order that was never filled is left out
    assert out["activity"][1] == {"date": "2026-11-16", "ticker": "AAA", "side": "buy", "qty": 10.0, "price": 5900.0, "dollars": 59000.0, "reason": "new in target"}


def test_after_the_close_the_day_comes_from_the_account_and_the_daily_bar():
    out = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake(clock_time="2026-11-17T16:05:00-05:00", is_open=False, equity="100900"))
    assert out["series"]["dates"][-1] == "2026-11-17" and out["series"]["p"][-1] == 1.009 and out["series"]["s"][-1] == round(1.02 / K, 6)


def test_recorded_closes_win_and_survive_a_failure_of_the_history():
    start = {"date": "2026-11-16", "at": "2026-11-16T15:00:30Z", "benchmark": 703.5, "k": round(K, 8)}
    ctx = {**CONTEXT, "baseline": start, "closes": {"2026-11-16": {"p": 1.0049, "s": 1.0051}}}       # stored: already counted from the purchases
    out = snapshot.build("k", "s", LAUNCH, ctx, get=fake(fail=("portfolio/history",)))
    assert out["series"]["p"][:2] == [1.0, 1.0049] and out["series"]["s"][:2] == [1.0, 1.0051] and len(out["series"]["dates"]) == 3
    out = snapshot.build("k", "s", LAUNCH, ctx, get=fake(fail=("portfolio/history", "/v2/stocks")))     # no benchmark at all: recorded days only
    assert out["series"]["dates"] == ["2026-11-16", "2026-11-16"]
    old = {**CONTEXT, "closes": {"2026-11-13": {"p": 1.0, "s": 1.0}, "2026-11-16": {"p": 1.0049, "s": 1.0101}}}   # written before the change
    out = snapshot.build("k", "s", LAUNCH, old, get=fake(fail=("portfolio/history",)))
    assert out["series"]["s"][1] == round(1.0101 / K, 6)             # recounted from the purchases until the close stores the new start


def test_same_picture_ignores_the_time_of_the_snapshot():
    a = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake(), now=datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc))
    b = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake(), now=datetime(2026, 11, 17, 16, 15, tzinfo=timezone.utc))
    c = snapshot.build("k", "s", LAUNCH, CONTEXT, get=fake(equity="101500"))
    assert a["updated"] != b["updated"] and snapshot.same_picture(a, b) and not snapshot.same_picture(a, c)


def test_dust_left_by_a_sale_is_not_a_holding():
    base = fake()
    def get(url, params, key, secret):
        out = base(url, params, key, secret)
        if url.endswith("/v2/positions"):
            out = out + [{"symbol": "ZZZ", "qty": "0.000000234", "market_value": "0.00006", "current_price": "262.1", "avg_entry_price": "260", "unrealized_pl": "0", "change_today": "0"}]
        if url.endswith("/v2/orders"):
            out = [{"id": "o9", "symbol": "ZZZ", "side": "sell", "filled_qty": "0.000000234", "filled_avg_price": "262.1", "filled_at": "2026-11-17T15:00:00Z", "submitted_at": ""}] + out
        return out
    blob = snapshot.build("k", "s", LAUNCH, CONTEXT, get=get)
    assert [p["ticker"] for p in blob["positions"]] == ["AAA", "BBB"] and [o["ticker"] for o in blob["activity"]] == ["BBB", "AAA"]
