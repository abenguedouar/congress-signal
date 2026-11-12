"""Daily prices, cached per ticker.

Two series are kept for each ticker:
* ``adj``   total-return price (splits and dividends), used for all returns;
* ``close`` split-adjusted close, used with split-adjusted share counts to get
  a market capitalisation that is correct on any past date.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


class _Collect(logging.Handler):
    """Keeps what yfinance logs during one download (that is where it reports a rate limit)."""
    def __init__(self):
        super().__init__()
        self.text: list[str] = []

    def emit(self, record):
        self.text.append(record.getMessage())


def _yf_batch(batch: list[str], start: str) -> tuple[dict[str, pd.DataFrame], bool]:
    """One request for a group of tickers. Returns (frames, rate_limited)."""
    import yfinance as yf
    out, seen, log = {}, _Collect(), logging.getLogger("yfinance")
    log.addHandler(seen)
    try:
        raw = yf.download([yahoo_symbol(t) for t in batch], start=start, auto_adjust=False, actions=True,
                          group_by="ticker", progress=False, threads=False)
    finally:
        log.removeHandler(seen)
    said = " ".join(seen.text)
    limited = "Rate limit" in said or "Too Many Requests" in said
    for t in batch:
        try:
            d = raw[yahoo_symbol(t)].dropna(subset=["Close"])
        except KeyError:
            continue
        if d.empty:
            continue
        out[t] = pd.DataFrame({"adj": d["Adj Close"], "close": d["Close"],
                               "split": d.get("Stock Splits", 0.0)})
    return out, limited


def _from_yfinance(tickers: list[str], start: str, size: int = 25, budget_minutes: float = 100.0) -> dict[str, pd.DataFrame]:
    """Yahoo refuses clients that ask too fast: small groups, and a growing wait whenever it says so."""
    out, wait, stop = {}, 60.0, time.time() + budget_minutes * 60
    for i in range(0, len(tickers), size):
        batch = tickers[i:i + size]
        for attempt in range(6):
            if time.time() > stop:
                print(f"[prices] time budget used up at ticker {i} of {len(tickers)}")
                return out
            try:
                got, limited = _yf_batch([t for t in batch if t not in out], start)
            except Exception as exc:
                print(f"[prices] batch failed: {type(exc).__name__}: {exc}")
                got, limited = {}, True
            out.update(got)
            if not limited:
                wait = max(60.0, wait / 2)
                break
            print(f"[prices] rate limited at ticker {i}; waiting {wait:.0f}s")
            time.sleep(wait)
            wait = min(wait * 2, 600.0)
        time.sleep(1.5)
    return out


ALPACA_BARS = "https://data.alpaca.markets/v2/stocks/bars"


class AlpacaRefused(RuntimeError):
    pass


def _alpaca_bars(ticker: str, asof: str, adjustment: str, start: str, end: str, key: str, secret: str) -> pd.Series:
    """Daily closes from Alpaca's market-data API. ``asof`` tells which company the symbol meant on that date,
    so a ticker that was later renamed or delisted is still found."""
    rows, token = [], None
    while True:
        params = {"symbols": ticker, "timeframe": "1Day", "start": start, "end": end, "limit": 10000,
                  "adjustment": adjustment, "feed": "sip", "asof": asof}
        if token:
            params["page_token"] = token
        r = requests.get(ALPACA_BARS, params=params, timeout=30,
                         headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        if r.status_code == 429:
            time.sleep(10)
            continue
        if r.status_code in (401, 403):
            raise AlpacaRefused(f"{r.status_code} {r.text[:200]}")
        if r.status_code != 200:
            return pd.Series(dtype=float)
        body = r.json()
        for part in (body.get("bars") or {}).values():   # keyed by the symbol in use at each period
            rows.extend(part)
        token = body.get("next_page_token")
        time.sleep(0.35)                                  # 200 requests per minute allowed
        if not token:
            break
    if not rows:
        return pd.Series(dtype=float)
    out = pd.Series({pd.Timestamp(x["t"][:10]): float(x["c"]) for x in rows}).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _from_alpaca(ticker: str, asof, key: str, secret: str, start: str = "2016-01-01") -> pd.DataFrame | None:
    """Second source, used for the tickers Yahoo no longer knows (renamed or delisted). History starts in 2016."""
    asof = str(max(pd.Timestamp(asof), pd.Timestamp("2016-01-04")).date())
    end = str((pd.Timestamp.now() - pd.Timedelta(days=1)).date())
    adj = _alpaca_bars(ticker, asof, "all", start, end, key, secret)
    if len(adj) < 20:
        return None
    close = _alpaca_bars(ticker, asof, "split", start, end, key, secret).reindex(adj.index).fillna(adj)
    return pd.DataFrame({"adj": adj, "close": close, "split": 0.0})


def update_prices(tickers: list[str], cache_dir: str | Path, start: str = "2011-01-01", max_age_days: int = 1,
                  asof: dict | None = None) -> dict[str, str]:
    """Refresh the cache; returns {ticker: status}.

    ``asof`` gives, per ticker, a date on which the symbol was in use (a date it was traded in a filing).
    A series that ended long ago (delisted stock) is never asked again; a ticker that no source knows is
    asked again only after three weeks.
    """
    cache = Path(cache_dir) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    now, today = pd.Timestamp.now(), str(pd.Timestamp.now().date())

    def stale(t: str) -> bool:
        m, f = meta.get(t, {}), cache / f"{t}.parquet"
        if not f.exists():
            return not (m.get("src") == "missing" and (now - pd.Timestamp(m["asked"])).days < 21)
        if m.get("last") and (pd.Timestamp(m["asked"]) - pd.Timestamp(m["last"])).days > 30:
            return False                                  # the series ended: nothing new will come
        return (now - pd.Timestamp(f.stat().st_mtime, unit="s")).days >= max_age_days

    todo = [t for t in tickers if stale(t)]
    status = {t: ("cached" if (cache / f"{t}.parquet").exists() else "missing") for t in tickers if t not in todo}
    got = {}
    try:
        got = _from_yfinance(todo, start)
    except Exception as exc:
        print(f"[prices] yfinance failed: {exc}")
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    use_alpaca, shown = bool(key and secret), 0
    for t in todo:
        d, src = got.get(t), "yfinance"
        if (d is None or d.empty) and use_alpaca:
            try:
                d, src = _from_alpaca(t, (asof or {}).get(t, "2016-01-04"), key, secret), "alpaca"
                if d is not None and shown < 5:
                    shown += 1
                    print(f"[prices] alpaca {t}: {len(d)} days, {d.index.min().date()} to {d.index.max().date()}")
            except AlpacaRefused as exc:
                print(f"[prices] alpaca refused the request ({exc}); second source off for this run")
                d, use_alpaca = None, False
            except Exception as exc:
                print(f"[prices] alpaca {t}: {type(exc).__name__}: {exc}")
                d = None
        if d is None or d.empty:
            status[t] = "missing"
            meta[t] = {"src": "missing", "asked": today}
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d.to_parquet(cache / f"{t}.parquet")
        status[t] = src
        meta[t] = {"src": src, "asked": today, "last": str(d.index.max().date())}
    meta_path.write_text(json.dumps(meta))
    return status


def load_prices(tickers: list[str], cache_dir: str | Path, field: str = "adj") -> pd.DataFrame:
    """Wide frame date x ticker for one field ('adj' or 'close')."""
    cache = Path(cache_dir) / "prices"
    cols = {}
    for t in tickers:
        p = cache / f"{t}.parquet"
        if p.exists():
            cols[t] = pd.read_parquet(p)[field]
    return pd.DataFrame(cols).sort_index()


@functools.lru_cache(maxsize=8)
def _splits(path: str) -> pd.Series:
    s = pd.read_parquet(path)["split"]
    return s[s > 0]


def split_factor_after(cache_dir: str | Path, ticker: str, date) -> float:
    """Cumulative split ratio after ``date``: converts shares reported then into today's share units."""
    p = Path(cache_dir) / "prices" / f"{ticker}.parquet"
    if not p.exists():
        return 1.0
    s = _splits(str(p))
    s = s[s.index > pd.Timestamp(date)]
    return float(s.prod()) if len(s) else 1.0


def clean_prices(prices: pd.DataFrame, lo: float = -0.9, hi: float = 3.0, max_bad: int = 2) -> tuple[pd.DataFrame, dict]:
    """Remove impossible quotes from a price panel (date x ticker).

    Free price sources carry broken adjusted series for some thinly traded
    lines: negative prices, one-day moves of several thousand percent. A
    ticker with more than ``max_bad`` such days is dropped; isolated ones are
    neutralised (that day's return set to zero) and the series is rebuilt.
    """
    p = prices.where(prices > 0)
    r = p.pct_change(fill_method=None)
    bad = (r < lo) | (r > hi)
    n_bad = bad.sum()
    dropped = sorted(n_bad.index[n_bad > max_bad])
    fixed = sorted(n_bad.index[(n_bad > 0) & (n_bad <= max_bad)])
    p = p.drop(columns=dropped)
    for t in fixed:
        s = p[t]
        first = s.first_valid_index()
        rr = r[t].where(~bad[t], 0.0)
        level = (1 + rr.fillna(0.0)).cumprod()
        p[t] = (s[first] * level / level[first]).where(s.notna())
    return p, {"tickers_dropped_for_broken_prices": dropped, "tickers_with_isolated_fix": fixed}
