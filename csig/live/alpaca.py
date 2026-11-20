"""Thin client for Alpaca's trading API - TEST ACCOUNT ONLY.

The address of the real-money service does not appear anywhere in this
module: an order sent from here cannot reach a real account, whatever keys
are supplied. Moving to real money is a deliberate code change, not a setting.
"""
from __future__ import annotations

import os
import time

import requests

PAPER = "https://paper-api.alpaca.markets"


class Broker:
    def __init__(self, key: str | None = None, secret: str | None = None):
        self.key = key or os.environ.get("ALPACA_API_KEY")
        self.secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not (self.key and self.secret):
            raise RuntimeError("Alpaca keys are not set")

    def _call(self, method: str, path: str, **kw):
        for attempt in range(4):
            r = requests.request(method, PAPER + path, timeout=30, headers={
                "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}, **kw)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca {method} {path}: {r.status_code} {r.text[:300]}")
            return r.json() if r.text else {}
        raise RuntimeError(f"Alpaca {method} {path}: rate limited")

    def account(self) -> dict:
        a = self._call("GET", "/v2/account")
        return {"equity": float(a["equity"]), "cash": float(a["cash"]), "status": a.get("status"),
                "blocked": bool(a.get("trading_blocked") or a.get("account_blocked")),
                "shorting_enabled": bool(a.get("shorting_enabled")), "multiplier": float(a.get("multiplier") or 1)}

    def clock(self) -> dict:
        return self._call("GET", "/v2/clock")

    def positions(self) -> dict[str, dict]:
        return {p["symbol"]: {"qty": float(p["qty"]), "value": float(p["market_value"]), "price": float(p["current_price"]),
                              "cost": float(p.get("avg_entry_price") or 0.0), "pnl": float(p.get("unrealized_pl") or 0.0)}
                for p in self._call("GET", "/v2/positions")}

    def close(self, symbol: str) -> dict:
        """Sells the whole position, fractions included (the broker computes the exact quantity)."""
        return self._call("DELETE", f"/v2/positions/{symbol}")

    def configure(self, **settings) -> dict:
        """Account settings, e.g. no_shorting=True, max_margin_multiplier="1"."""
        return self._call("PATCH", "/v2/account/configurations", json=settings)

    def open_orders(self) -> list[dict]:
        return self._call("GET", "/v2/orders", params={"status": "open", "limit": 500})

    def tradable(self, symbol: str) -> bool:
        try:
            a = self._call("GET", f"/v2/assets/{symbol}")
        except RuntimeError:
            return False
        return bool(a.get("tradable") and a.get("fractionable") and a.get("status") == "active")

    def buy(self, symbol: str, dollars: float, tag: str) -> dict:
        """Market order for a dollar amount, valid for the day. Never on margin: the planner sizes it within the cash."""
        return self._call("POST", "/v2/orders", json={"symbol": symbol, "notional": f"{dollars:.2f}", "side": "buy",
                                                      "type": "market", "time_in_force": "day", "client_order_id": tag})

    def sell(self, symbol: str, qty: float, tag: str) -> dict:
        return self._call("POST", "/v2/orders", json={"symbol": symbol, "qty": f"{qty:.6f}", "side": "sell",
                                                      "type": "market", "time_in_force": "day", "client_order_id": tag})
