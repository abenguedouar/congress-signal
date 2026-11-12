"""Unified trade table.

Every source (official House PDFs, the Senate eFD site, public mirrors) is
converted to the same columns, listed in ``csig.config.TRADE_COLUMNS``.
The key column is ``disclosure_date``: the day the filing became public.
Nothing in the system may use a trade before that day.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..config import TRADE_COLUMNS

_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,4}([.\-][A-Z])?$")
_OWNER = {"SP": "spouse", "JT": "joint", "DC": "child", "DE": "child",
          "SPOUSE": "spouse", "JOINT": "joint", "CHILD": "child", "SELF": "self", "": "self"}
_SIDE = {"P": "buy", "PURCHASE": "buy", "BUY": "buy",
         "S": "sell", "S (PARTIAL)": "sell", "SALE": "sell", "SALE (FULL)": "sell",
         "SALE (PARTIAL)": "sell", "SELL": "sell", "E": "exchange", "EXCHANGE": "exchange"}


def parse_amount(text: str) -> tuple[float | None, float | None]:
    """'$1,001 - $15,000' -> (1001, 15000); 'Over $50,000,000' -> (50000000, None)."""
    nums = [float(n.replace(",", "")) for n in re.findall(r"\$?\s*([\d,]{3,})", text or "")]
    if not nums:
        return None, None
    if re.search(r"over", text, re.I) or len(nums) == 1:
        return nums[0], None
    return nums[0], nums[1]


def clean_ticker(raw: str | None) -> str | None:
    if not raw:
        return None
    t = re.sub(r"<[^>]+>", "", str(raw)).strip().upper().replace("/", ".")
    return t if _TICKER.match(t) and t not in {"N.A", "NA"} else None


def _asset_type(raw: str | None) -> str:
    r = (raw or "").strip().lower()
    if r in {"st", "stock", "stocks (including adrs)"}:
        return "stock"
    if r in {"op", "stock option", "options"}:
        return "option"
    return "other"


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Type, clean and de-duplicate a frame that already has TRADE_COLUMNS."""
    df = df.reindex(columns=TRADE_COLUMNS).copy()
    for c in ("trade_date", "disclosure_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["trade_date", "disclosure_date"])
    df = df[df["side"].isin(["buy", "sell", "exchange"])]
    df["id_src"] = df["id_src"].fillna("coded")          # sources with an explicit type column
    for c in ("note", "account"):                        # the filer's free-text description; the account that traded
        df[c] = df[c].fillna("").astype(str)
    df = df.drop_duplicates(subset=["member_id", "member_name", "ticker", "asset_name", "side",
                                    "trade_date", "disclosure_date", "amount_min"])
    return df.sort_values(["disclosure_date", "trade_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Public mirrors (used for development and as a cross-check on our parsers)
# ----------------------------------------------------------------------
def from_capitol_cache(path: str | Path) -> pd.DataFrame:
    """House trades as cached by the open-source capitol-api project."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["trades"]
    rows = []
    for t in raw:
        tx, a, p = t["transaction"], t["asset"], t.get("person") or {}
        rows.append({
            "trade_id": f"H-{t['id']}", "chamber": "house",
            "member_id": p.get("bioguideId"), "member_name": p.get("name"),
            # this mirror writes "DC" when the owner box is blank, i.e. the member himself
            "owner": {"SP": "spouse", "JT": "joint"}.get((t.get("owner") or "").upper(), "self"),
            "ticker": clean_ticker(a.get("ticker")), "asset_name": a.get("name"),
            "asset_type": _asset_type(a.get("type")), "side": tx.get("category"),
            "trade_date": tx.get("tradeDate"), "disclosure_date": tx.get("filingDate"),
            "amount_min": tx.get("amountMin"), "amount_max": tx.get("amountMax"),
            "source_url": t["filing"].get("pdfUrl"),
        })
    return finalize(pd.DataFrame(rows))


def from_senate_watcher(data_dir: str | Path, legislators=None) -> pd.DataFrame:
    """Senate trades from the senate-stock-watcher mirror (one file per filing day)."""
    rows = []
    for f in sorted(Path(data_dir).glob("transaction_report_for_*.json")):
        for rep in json.loads(f.read_text(encoding="utf-8")):
            disclosed = rep.get("date_recieved")
            first, last = rep.get("first_name", ""), rep.get("last_name", "")
            mid = None
            if legislators is not None:
                mid = legislators.match(first, last, "senate", pd.to_datetime(disclosed, errors="coerce"))
            for i, tx in enumerate(rep.get("transactions") or []):
                lo, hi = parse_amount(tx.get("amount", ""))
                rows.append({
                    "trade_id": f"S-{rep.get('ptr_link', '').rstrip('/').split('/')[-1]}-{i + 1}",
                    "chamber": "senate", "member_id": mid, "member_name": f"{first} {last}".strip(),
                    "owner": _OWNER.get((tx.get("owner") or "").upper(), "self"),
                    "ticker": clean_ticker(tx.get("ticker")), "asset_name": tx.get("asset_description"),
                    "asset_type": _asset_type(tx.get("asset_type")),
                    "side": _SIDE.get((tx.get("type") or "").upper()),
                    "trade_date": pd.to_datetime(tx.get("transaction_date"), format="%m/%d/%Y", errors="coerce"),
                    "disclosure_date": pd.to_datetime(disclosed, format="%m/%d/%Y", errors="coerce"),
                    "amount_min": lo, "amount_max": hi, "source_url": rep.get("ptr_link"),
                })
    return finalize(pd.DataFrame(rows))


# What filers write in the description box. "Mechanical" rows are not decisions (dividends reinvested,
# shares received in a spin-off, a merger, an inheritance); they are dropped from the signals.
# "Managed" rows come from accounts run by an adviser; they are kept and flagged.
_MECHANICAL = re.compile(r"reinvest|spin[- ]?off|merger|stock split|inherit|\bgift|conversion", re.I)
_MANAGED = re.compile(r"managed|rebalanc|discretion|\buma\b|advisor|adviser|automatic|parametric|tax[- ]loss|\btlh\b|"
                      r"blind trust|without my|not directed", re.I)


def stock_trades(df: pd.DataFrame, max_lag_days: int = 400, include_options: bool = True,
                 drop_mechanical: bool = True) -> pd.DataFrame:
    """Disclosed stock purchases and sales with a usable ticker and sane dates.

    Option trades are read as information about the underlying stock, never as
    something to trade: buying a call counts as a purchase signal, buying a put
    as a sale signal. Other option trades (writing, closing) are ambiguous and dropped.
    """
    df = df.copy()
    for c in ("note", "account"):
        df[c] = df[c].fillna("").astype(str) if c in df else ""
    df["via_option"] = 0.0
    if include_options:
        words = (df["asset_name"].fillna("") + " " + df["note"]).str.lower()
        opt = df["asset_type"].eq("option") & df["side"].eq("buy") & df["ticker"].notna()
        is_call, is_put = words.str.contains(r"\bcalls?\b"), words.str.contains(r"\bputs?\b")
        call, put = opt & is_call & ~is_put, opt & is_put & ~is_call
        df.loc[call | put, "via_option"] = 1.0
        df.loc[call | put, "asset_type"] = "stock"
        df.loc[put, "side"] = "sell"
    df["managed"] = (df["note"] + " " + df["account"]).str.contains(_MANAGED).astype(float)
    mechanical = df["note"].str.contains(_MECHANICAL)
    lag = (df["disclosure_date"] - df["trade_date"]).dt.days
    keep = (df["asset_type"].eq("stock") & df["ticker"].notna()
            & df["side"].isin(["buy", "sell"]) & lag.between(0, max_lag_days))
    if drop_mechanical:
        keep &= ~mechanical
    out = df[keep].copy()
    out["lag_days"] = lag[keep]
    out["amount_mid"] = out[["amount_min", "amount_max"]].mean(axis=1).fillna(out["amount_min"])
    return out.reset_index(drop=True)


def signals(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """``stock_trades`` with the project's settings; tickers of companies that only changed their symbol
    are brought to the current one (the symbol written in the filing is kept in ``ticker_filed``)."""
    ev = cfg["events"]
    st = stock_trades(trades, ev["max_lag_days"], ev.get("include_call_options", True))
    st["ticker_filed"] = st["ticker"]
    st["ticker"] = st["ticker"].replace((cfg.get("tickers") or {}).get("aliases") or {})
    return st
