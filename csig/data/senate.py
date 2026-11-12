"""Official Senate source: the Electronic Financial Disclosure site (eFD).

Flow: accept the access agreement once per session, query the report search
for Periodic Transaction Reports, then read each electronic report page.
Paper filings (scanned images) are logged as unparsed.

The site sometimes challenges automated clients. If that happens on the
runner, ``SenateBlocked`` is raised and the pipeline falls back to the
secondary source configured in ``scripts/build_dataset.py``.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .trades import _OWNER, _SIDE, clean_ticker, finalize, parse_amount

ROOT = "https://efdsearch.senate.gov"
HOME = f"{ROOT}/search/home/"
SEARCH = f"{ROOT}/search/"
DATA = f"{ROOT}/search/report/data/"
PTR_REPORT_TYPE = 11
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"}


class SenateBlocked(RuntimeError):
    pass


def open_session(attempts: int = 4, wait: float = 45.0) -> requests.Session:
    """Accept the access agreement. The site refuses some addresses for a while: retry before giving up."""
    last = ""
    for k in range(attempts):
        if k:
            time.sleep(wait * k)
        s = requests.Session()
        s.headers.update(UA)
        try:
            r = s.get(HOME, timeout=60)
            token = BeautifulSoup(r.text, "lxml").find("input", {"name": "csrfmiddlewaretoken"})
            if r.status_code != 200 or token is None:
                last = f"landing page status {r.status_code}"
                continue
            r = s.post(HOME, data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token["value"]},
                       headers={"Referer": HOME}, timeout=60)
            if r.status_code != 200 or "csrftoken" not in s.cookies:
                last = f"agreement status {r.status_code}"
                continue
            return s
        except requests.RequestException as exc:
            last = type(exc).__name__
    raise SenateBlocked(last)


def list_reports(s: requests.Session, since: str = "01/01/2012", page: int = 100, pause: float = 1.0) -> pd.DataFrame:
    """All PTR filings submitted since ``since`` (mm/dd/yyyy)."""
    rows, start = [], 0
    while True:
        r = s.post(DATA, timeout=60, headers={"Referer": SEARCH, "X-CSRFToken": s.cookies["csrftoken"]}, data={
            "start": str(start), "length": str(page), "report_types": f"[{PTR_REPORT_TYPE}]",
            "filer_types": "[]", "submitted_start_date": f"{since} 00:00:00", "submitted_end_date": "",
            "candidate_state": "", "senator_state": "", "office_id": "", "first_name": "", "last_name": "",
        })
        if r.status_code != 200:
            raise SenateBlocked(f"search status {r.status_code}")
        batch = r.json().get("data", [])
        for first, last, _office, link_html, date in batch:
            a = BeautifulSoup(link_html, "lxml").find("a")
            rows.append({"first": first.strip(), "last": last.strip(), "url": ROOT + a["href"],
                         "paper": "/paper/" in a["href"],
                         "disclosure_date": pd.to_datetime(date, format="%m/%d/%Y", errors="coerce")})
        if len(batch) < page:
            break
        start += page
        time.sleep(pause)
    return pd.DataFrame(rows)


def parse_report(html: str) -> list[dict]:
    """Rows of the transaction table of one electronic PTR page."""
    table = BeautifulSoup(html, "lxml").find("table")
    out = []
    if table is None:
        return out
    for tr in table.find_all("tr")[1:]:
        cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if len(cells) < 8:
            continue
        _, traded, owner, ticker, asset, atype, tx, amount = cells[:8]
        lo, hi = parse_amount(amount)
        out.append({
            "owner": _OWNER.get(owner.upper(), "self"), "ticker": clean_ticker(ticker),
            "asset_name": asset, "asset_type": "stock" if atype.lower() == "stock" else
            ("option" if "option" in atype.lower() else "other"),
            "side": _SIDE.get(tx.upper()),
            "trade_date": pd.to_datetime(traded, format="%m/%d/%Y", errors="coerce"),
            "amount_min": lo, "amount_max": hi,
            "note": cells[8][:300] if len(cells) > 8 and cells[8] != "--" else "",
        })
    return out


def ingest(since: str = "01/01/2012", legislators=None, known_urls: set[str] | None = None,
           pause: float = 0.8, limit: int | None = None, deadline: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = open_session()
    reports = list_reports(s, since, pause=pause).sort_values("disclosure_date", ascending=False)
    trades, log, fails = [], [], 0
    for _, rep in reports.iterrows():
        if known_urls and rep["url"] in known_urls:
            continue
        if (limit is not None and len(log) >= limit) or (deadline is not None and time.time() > deadline):
            break
        if fails >= 8:                                   # the site stopped answering: keep what we have
            print(f"[senate] stopped after {fails} failures in a row")
            break
        if rep["paper"]:
            log.append({"url": rep["url"], "status": "paper", "n": 0})
            continue
        try:
            r = s.get(rep["url"], headers={"Referer": SEARCH}, timeout=60)
            if r.status_code in (403, 429, 503):         # throttled: wait, renew the session, try once more
                time.sleep(20)
                s = open_session(attempts=2)
                r = s.get(rep["url"], headers={"Referer": SEARCH}, timeout=60)
        except (SenateBlocked, requests.RequestException) as exc:
            log.append({"url": rep["url"], "status": f"error:{type(exc).__name__}", "n": 0})
            fails += 1
            continue
        if r.status_code != 200:
            log.append({"url": rep["url"], "status": f"http_{r.status_code}", "n": 0})
            fails += 1
            continue
        fails = 0
        rows = parse_report(r.text)
        mid = legislators.match(rep["first"], rep["last"], "senate", rep["disclosure_date"]) if legislators else None
        uid = rep["url"].rstrip("/").split("/")[-1]
        for i, row in enumerate(rows):
            row.update({"trade_id": f"S-{uid}-{i + 1}", "chamber": "senate", "member_id": mid,
                        "member_name": f"{rep['first']} {rep['last']}".strip(),
                        "disclosure_date": rep["disclosure_date"], "source_url": rep["url"]})
        trades.extend(rows)
        log.append({"url": rep["url"], "status": "parsed" if rows else "no_rows", "n": len(rows)})
        time.sleep(pause)
    return finalize(pd.DataFrame(trades)) if trades else pd.DataFrame(), pd.DataFrame(log)
