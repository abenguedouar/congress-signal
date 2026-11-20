"""What the page needs beside the account itself: company names, the members of Congress whose disclosed
purchases explain each holding, and the reason of each order. Written at every daily decision."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def company_name(raw: str, ticker: str) -> str:
    """'RPM INTERNATIONAL INC/DE/' -> 'RPM International Inc.' (the regulator's register often writes names in capitals)."""
    raw = re.sub(r"\s*/[A-Za-z]{2,3}/?\s*$", "", raw.strip())
    letters = [c for c in raw if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) > 0.7 * len(letters):          # written in capitals in the register
        words = []
        for w in raw.split():
            bare = w.strip(".,").upper()
            keep = bare == ticker.upper() or bare in ("USA", "US", "II", "III", "IV") or (len(bare) <= 3 and not re.search("[AEIOUY]", bare))
            words.append(w.upper() if keep else w.capitalize())
        raw = " ".join(words)
    out = re.sub(r"\b(\w+) Com\b", r"\1.com", raw)
    return re.sub(r"\b(Inc|Corp|Co|Ltd)$", r"\1.", out)


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def build_context(cfg: dict, store: Path, tickers, active: pd.DataFrame, ledger: pd.DataFrame | None = None) -> dict:
    """``tickers``: target and holdings. ``active``: the signals behind the target (csig.live.target.target_weights)."""
    names = {}
    companies = Path(store) / "panel" / "companies.json"
    if companies.exists():
        raw = json.loads(companies.read_text())
        names = {t: company_name(raw[t].get("name") or t, t) for t in tickers if t in raw}
    who = {}
    if len(active):
        for t, g in active.sort_values("disclosed", ascending=False).groupby("ticker", sort=False):
            who[t] = [{"member": r.member, "chamber": r.chamber or "", "disclosed": str(pd.Timestamp(r.disclosed).date()),
                       "traded": "" if pd.isna(r.traded) else str(pd.Timestamp(r.traded).date()),
                       "lo": _num(r.amount_min), "hi": _num(r.amount_max), "owner": r.owner or ""} for r in g.itertuples()]
    reasons = {}
    if ledger is not None and len(ledger) and "order_id" in ledger:
        reasons = {str(i): r for i, r in zip(ledger["order_id"], ledger["reason"]) if i}
    return {"launch": str(cfg["live"]["launch_date"]), "base_equity": float(cfg["live"].get("initial_equity", 100_000.0)),
            "names": names, "who": who, "reasons": reasons}


def read_context(live: Path) -> dict:
    """Context, recorded closes and the list of statements, as one dictionary for csig.live.snapshot.build."""
    out = {}
    for name, key in (("context.json", None), ("closes.json", "closes"), ("reports.json", "reports"), ("baseline.json", "baseline")):
        f = Path(live) / name
        if f.exists():
            data = json.loads(f.read_text())
            out.update(data if key is None else {key: data})
    return out
