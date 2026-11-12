"""Company information from SEC EDGAR: ticker list, name, sector code, state.

``company`` can also keep a chosen set of the facts a company filed (XBRL
"company facts"); which ones is decided by the caller.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

SEC_UA = os.environ.get("SEC_USER_AGENT") or "congress-signal research (set SEC_USER_AGENT)"


def _get(url: str) -> dict:
    r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    time.sleep(0.12)  # SEC fair-access limit: 10 requests per second
    return r.json()


def ticker_map(cache_dir: str | Path) -> dict[str, int]:
    p = Path(cache_dir) / "sec" / "company_tickers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(_get("https://www.sec.gov/files/company_tickers.json")))
    raw = json.loads(p.read_text())
    return {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in raw.values()}


def company(cik: int, cache_dir: str | Path, keep: set | None = None, version: int = 0, refresh: bool = False) -> dict:
    """{'name', 'sic', 'sic_description', 'state', 'facts'} for one company, cached on disk.

    ``keep`` is a set of (taxonomy, tag): those facts are downloaded and stored as filed. Without it only the
    company's description is read. ``version`` identifies the set; a cached file with another version is replaced.
    """
    p = Path(cache_dir) / "sec" / f"{cik:010d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not refresh:
        got = json.loads(p.read_text())
        if keep is None or got.get("v") == version:
            return got
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    facts = {}
    if keep is not None:
        try:
            facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").get("facts", {})
        except requests.HTTPError:
            facts = {}
        facts = {taxo: {tag: body for tag, body in tags.items() if (taxo, tag) in keep} for taxo, tags in facts.items()}
    out = {"v": version if keep is not None else 0, "cik": cik, "name": sub.get("name"), "sic": str(sub.get("sic") or ""),
           "sic_description": sub.get("sicDescription") or "",
           "state": ((sub.get("addresses") or {}).get("business") or {}).get("stateOrCountry") or "",
           "facts": facts}
    p.write_text(json.dumps(out))
    return out
