"""The state of the test account, as the public page shows it: value, holdings, daily series against the
benchmark, orders, and whether the market is open.

Standard library only, so that the job which refreshes the page during the session installs nothing.
TEST ACCOUNT ONLY (the address of the real-money service appears nowhere), and read-only: nothing here sends an order.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
BENCHMARK = "SPY"
DUST = 1.0          # dollars: what is left of a position after a sale rounded to the millionth of a share is not a holding


def http_get(url: str, params: dict, key: str, secret: str, attempts: int = 3):
    query = url + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(query, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"})
    for k in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and k < attempts - 1:
                time.sleep(2 * (k + 1))
                continue
            raise RuntimeError(f"{url}: {exc.code} {exc.read()[:200]!r}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if k < attempts - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"{url}: {exc}") from None


def waiting(launch: str, base_equity: float = 100_000.0, now: datetime | None = None) -> dict:
    """What the page shows before the account has anything to report."""
    now = now or datetime.now(timezone.utc)
    return {"mode": "waiting", "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "launch": launch, "market": {"open": False},
            "equity": base_equity, "cash": base_equity, "day": 0.0, "positions": [],
            "series": {"dates": [], "p": [], "s": []}, "activity": [], "reports": []}


def _benchmark(call, launch: str, now: datetime) -> tuple[dict, float | None]:
    """({day: close adjusted for dividends}, latest price or None). Closes up to 20 minutes ago are free on the full feed."""
    start = (date.fromisoformat(launch) - timedelta(days=10)).isoformat()
    closes = {}
    for feed in ("sip", "iex"):
        try:
            params = {"symbols": BENCHMARK, "timeframe": "1Day", "start": start, "adjustment": "all", "feed": feed, "limit": 10000}
            if feed == "sip":
                params["end"] = (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
            body = call(DATA, "/v2/stocks/bars", **params)
            closes = {b["t"][:10]: float(b["c"]) for part in (body.get("bars") or {}).values() for b in part}
            if closes:
                break
        except RuntimeError:
            continue
    latest = None
    try:
        snap = call(DATA, f"/v2/stocks/{BENCHMARK}/snapshot", feed="iex")
        bar = snap.get("dailyBar") or {}
        if bar.get("t") and bar["t"][:10] not in closes:            # the session in progress, or closed minutes ago
            closes[bar["t"][:10]] = float(bar["c"])
        latest = float((snap.get("latestTrade") or {}).get("p") or 0) or None
    except RuntimeError:
        pass
    return closes, latest


def _orders(call, launch: str) -> list[dict]:
    """Filled orders since the launch, most recent first."""
    out, until = [], None
    for _ in range(10):
        params = {"status": "closed", "limit": 500, "after": launch + "T00:00:00Z", "direction": "desc"}
        if until:
            params["until"] = until
        page = call(PAPER, "/v2/orders", **params) or []
        out += page
        if len(page) < 500:
            break
        until = page[-1]["submitted_at"]
    return [o for o in out if float(o.get("filled_qty") or 0) > 0]


def purchase_baseline(call, orders: list[dict], close_before: float) -> dict | None:
    """When the first purchases were filled, and the benchmark at that minute relative to its last close before.
    The portfolio is cash until then: measuring both from that minute compares like with like."""
    fills = sorted(o["filled_at"] for o in orders if o.get("filled_at") and o.get("side") == "buy")
    if not fills or not close_before:
        return None
    start = datetime.fromisoformat(fills[0][:16] + ":00+00:00")
    for feed in ("sip", "iex"):
        try:
            body = call(DATA, "/v2/stocks/bars", symbols=BENCHMARK, timeframe="1Min", start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=(start + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"), adjustment="raw", feed=feed, limit=10)
            bars = [b for part in (body.get("bars") or {}).values() for b in part]
            if bars:
                price = float(bars[0].get("vw") or bars[0]["c"])
                return {"date": fills[0][:10], "at": fills[0][:19] + "Z", "benchmark": price, "k": round(price / close_before, 8)}
        except RuntimeError:
            continue
    return None


def build(key: str, secret: str, launch: str, context: dict | None = None, get=http_get, now: datetime | None = None) -> dict:
    """``context`` (written by the daily decision and the close): names, who (members behind each stock),
    reasons (order id -> why), closes ({day: {"p", "s"}} recorded at each close), reports, base_equity, and
    baseline (the first purchases; once it is stored, the recorded closes are relative to it)."""
    context = context or {}
    now = now or datetime.now(timezone.utc)
    call = lambda base, path, **params: get(base + path, params, key, secret)
    base = float(context.get("base_equity") or 100_000.0)
    account, clock = call(PAPER, "/v2/account"), call(PAPER, "/v2/clock")
    today = clock["timestamp"][:10]                                   # the exchange's own date
    if today < launch:
        out = waiting(launch, base, now)
        out["market"] = {"open": bool(clock.get("is_open")), "next_open": clock.get("next_open"), "next_close": clock.get("next_close")}
        return out
    is_open = bool(clock.get("is_open"))
    equity, cash = float(account["equity"]), float(account["cash"])
    last_equity = float(account.get("last_equity") or 0)

    names, who = context.get("names", {}), context.get("who", {})
    positions = [{"ticker": p["symbol"], "name": names.get(p["symbol"], p["symbol"]), "qty": float(p["qty"]),
                  "price": float(p["current_price"]), "value": float(p["market_value"]), "weight": float(p["market_value"]) / equity if equity else 0.0,
                  "cost": float(p.get("avg_entry_price") or 0), "pnl": float(p.get("unrealized_pl") or 0),
                  "day": float(p.get("change_today") or 0), "who": who.get(p["symbol"], [])}
                 for p in call(PAPER, "/v2/positions") if abs(float(p["market_value"])) >= DUST]
    positions.sort(key=lambda p: -p["value"])

    # ---- daily series: 1 on the last close before the launch
    spy, latest = _benchmark(call, launch, now)
    closes = context.get("closes") or {}
    recorded = {d: v for d, v in closes.items() if d >= launch}
    values = {}
    try:
        start = (date.fromisoformat(launch) - timedelta(days=10)).isoformat() + "T00:00:00Z"
        hist = call(PAPER, "/v2/account/portfolio/history", timeframe="1D", start=start)
        for ts, v in zip(hist.get("timestamp") or [], hist.get("equity") or []):
            day = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
            if v and launch <= day < today:
                values[day] = float(v) / base
    except RuntimeError:
        pass
    if is_open and latest:
        spy[today] = latest
    d0 = max([d for d in list(spy) + list(closes) if d < launch], default=None)     # the last close before the launch
    orders = _orders(call, launch)
    stored = context.get("baseline")
    fresh = None if stored or not d0 or d0 not in spy else purchase_baseline(call, orders, spy[d0])
    start = stored or fresh                        # None until the first purchase: the series then starts at the close of d0
    k = float(start["k"]) if start else 1.0
    dates, p, s = [], [], []
    if d0 or start:
        dates, p, s = [start["date"] if start else d0], [1.0], [1.0]
        for d in sorted(set(spy) | set(recorded)):
            if d < (start["date"] if start else launch) or d > today:
                continue
            if d in recorded and not (d == today and is_open):       # what was written down at that day's close
                pv, sv = recorded[d]["p"], recorded[d]["s"] / (float(fresh["k"]) if fresh else 1.0)
            elif d in spy and d0 in spy and (d in values or d == today):
                pv, sv = (equity / base if d == today else values[d]), spy[d] / spy[d0] / k
            else:
                continue
            dates.append(d); p.append(round(pv, 6)); s.append(round(sv, 6))

    reasons = context.get("reasons", {})
    activity = []
    for o in orders[:300]:
        qty, price = float(o["filled_qty"]), float(o.get("filled_avg_price") or 0)
        if qty * price < DUST:                                    # the sale of leftover dust is not an order worth showing
            continue
        activity.append({"date": (o.get("filled_at") or o["submitted_at"])[:10], "ticker": o["symbol"], "side": o["side"],
                         "qty": qty, "price": price, "dollars": round(qty * price, 2), "reason": reasons.get(o.get("id"), "")})

    return {"mode": "live", "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "launch": launch,
            "market": {"open": is_open, "next_open": clock.get("next_open"), "next_close": clock.get("next_close")},
            "equity": equity, "cash": cash, "day": (equity / last_equity - 1) if last_equity else 0.0,
            "positions": positions, "series": {"dates": dates, "p": p, "s": s}, "activity": activity,
            "reports": context.get("reports", []), "baseline": start}


def same_picture(a: dict, b: dict) -> bool:
    """True when two snapshots would draw the same page (the time of the snapshot aside)."""
    strip = lambda x: {k: v for k, v in x.items() if k != "updated"}
    return strip(a) == strip(b)
