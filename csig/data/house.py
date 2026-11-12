"""Official House source: Clerk of the House financial disclosures.

* yearly index  https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip
* one PDF per Periodic Transaction Report (PTR)
  https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf

Electronic PTRs are text PDFs. Their text is extracted once and kept in
``data/store/house_text`` (one small file per filing month), and the trade
table is rebuilt from that text on every run: a parser fix never needs a new
download. Paper filings are scans: they are logged so the gap is visible.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

from .assets import interpret
from .trades import _SIDE, finalize, parse_amount

BASE = "https://disclosures-clerk.house.gov/public_disc"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; congress disclosure study)"}

_DATE = r"\d{1,2}/\d{1,2}/\d{4}"
# A transaction row is recognised by its "data" part:  P 12/05/2025 12/08/2025 $100,001 - [$250,000]
DATA = re.compile(
    rf"^(?P<pre>.*?)(?:^|\s)(?P<tx>S\s*\(partial\)|P|S|E)\s*(?P<traded>{_DATE})\s*(?P<notified>{_DATE})\s*"
    rf"(?P<amount>(?:Spouse/DC\s+)?Over(?:\s*\$[\d,]+)?|\$[\d,]+(?:\s*-\s*(?:\$[\d,]+)?)?)", re.I)
# Lines that end a row: form labels ("F S : New", "S O : ...", "D : ..."), section titles, boilerplate.
BOUNDARY = re.compile(
    r"^(?:[A-Za-z](?:\s+[A-Za-z])*\s*:|(?:[A-Z]\s+){1,}[A-Z]?\s*$|[A-Z]\s*$|filing\s*id|id\s+owner|type\s+date|\$200\?|"
    r"name\s*:|status\s*:|state/district|clerk of|\* for the|yes\s+no|i certify|my knowledge|digitally signed|"
    r"filing status|sub-?holding|description|location|comments?\s*:)", re.I)
# "Description: text", "D : text", "Subholding of: text", "S O : text", "Filing status: New", "L : US", "C : text"
LABEL = re.compile(r"^(?P<label>[A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s*:\s*(?P<value>.*)$")
SECTION = re.compile(r"^(?:asset class details|initial public offerings|certification and signature|transactions$|"
                     r"filer information|\* for the|i certify|digitally signed|yes\s+no|nmlkj|gfedc|id\s+owner|type\s+date|\$200\?)", re.I)
TEXT_COLUMNS = ["doc_id", "year", "first", "last", "filing_date", "text"]


def fetch_index(year: int, session: requests.Session | None = None) -> pd.DataFrame:
    """All filings of a year; PTRs have FilingType == 'P'."""
    s = session or requests.Session()
    r = s.get(f"{BASE}/financial-pdfs/{year}FD.zip", headers=HEADERS, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        xml_name = next(n for n in z.namelist() if n.lower().endswith(".xml"))
        root = ElementTree.fromstring(z.read(xml_name))
    rows = [{c.tag: (c.text or "").strip() for c in m} for m in root.iter("Member")]
    df = pd.DataFrame(rows)
    df["FilingDate"] = pd.to_datetime(df["FilingDate"], format="%m/%d/%Y", errors="coerce")
    return df


def parse_ptr_text(text: str) -> list[dict]:
    """Extract transactions from the text layer of one PTR PDF.

    Layout of the live site: the first line of a row carries the start of the
    asset name, the transaction type, both dates and the amount; when the name
    or the amount wraps, the rest (often with the ticker and the asset-type
    code) sits on the following lines, up to the "Filing Status" label. When
    the data line carries no name at all, the name is the block just above it.
    """
    lines = [ln.strip() for ln in text.replace("\x00", "").split("\n")]
    hits = [(k, DATA.match(ln)) for k, ln in enumerate(lines)]
    hits = [(k, m) for k, m in hits if m]
    rows = []
    for n, (k, m) in enumerate(hits):
        stop = hits[n + 1][0] if n + 1 < len(hits) else len(lines)
        after, labelled, field = [], {"d": [], "s": []}, None
        for ln in lines[k + 1:stop]:
            if not ln or SECTION.match(ln):
                if ln:
                    break
                continue
            lab = LABEL.match(ln)
            if lab:                                      # "Description: ...", "Subholding of: ...", "Filing status: ..."
                words = lab.group("label").lower().split()
                field = "d" if words[0][0] in "dc" else ("s" if words[0][0] == "s" and len(words) > 1 else "x")
                if field in labelled:
                    labelled[field].append(lab.group("value"))
            elif BOUNDARY.match(ln):
                break
            elif field is None:
                after.append(ln)                         # the asset name or the amount wrapped
            elif field in labelled:
                labelled[field].append(ln)               # a description that runs over several lines
        pre, before = m.group("pre").strip(), []
        if not re.search(r"[A-Za-z]{3}", pre):           # no name on the data line: it is printed above
            j = k - 1
            while j >= 0 and lines[j] and not BOUNDARY.match(lines[j]) and not DATA.match(lines[j]) and len(before) < 4:
                before.insert(0, lines[j])
                j -= 1
        amount = m.group("amount")
        if amount.rstrip().endswith("-") or "$" not in amount:   # the figure wrapped onto the next line
            more = re.search(r"\$[\d,]+", " ".join(after))
            amount += " " + (more.group(0) if more else "")
        cell = " ".join([pre] + before + [re.sub(r"\$[\d,]+", "", a) for a in after])   # pre: owner code or empty when the name is above
        got = interpret(cell)
        lo, hi = parse_amount(amount)
        rows.append({
            "owner": got["owner"] or "self", "ticker": got["ticker"], "asset_name": got["asset_name"],
            "asset_type": got["asset_type"], "id_src": got["id_src"],
            "side": _SIDE.get(re.sub(r"\s+", " ", m.group("tx").upper()).replace("S(PARTIAL)", "S (PARTIAL)")),
            "trade_date": pd.to_datetime(m.group("traded"), format="%m/%d/%Y", errors="coerce"),
            "notification_date": pd.to_datetime(m.group("notified"), format="%m/%d/%Y", errors="coerce"),
            "amount_min": lo, "amount_max": hi,
            "note": " ".join(labelled["d"]).strip()[:300], "account": " ".join(labelled["s"]).strip()[:120],
        })
    return rows


def pdf_text(content: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# ----------------------------------------------------------------------
# Text store
# ----------------------------------------------------------------------
def load_texts(store_dir: str | Path) -> pd.DataFrame:
    files = sorted((Path(store_dir) / "house_text").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=TEXT_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def save_texts(texts: pd.DataFrame, store_dir: str | Path, only_months: set[str] | None = None) -> None:
    """One file per filing month, so a daily update rewrites a few kilobytes."""
    out = Path(store_dir) / "house_text"
    out.mkdir(parents=True, exist_ok=True)
    month = pd.to_datetime(texts["filing_date"]).dt.strftime("%Y-%m")
    for key, part in texts.groupby(month):
        if only_months is None or key in only_months:
            part.sort_values("doc_id").reset_index(drop=True).to_parquet(out / f"{key}.parquet", compression="zstd")


def fetch_year(year: int, known_docs: set[str] | None = None, pause: float = 0.25, limit: int | None = None,
               deadline: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the new PTRs of a year and extract their text.

    Returns (texts, log). ``log`` has one row per filing that gave no text:
    scans and download failures.
    """
    s = requests.Session()
    idx = fetch_index(year, s)
    ptr = idx[idx["FilingType"] == "P"].sort_values("FilingDate")
    texts, log = [], []
    for _, f in ptr.iterrows():
        doc = f["DocID"]
        if known_docs and doc in known_docs:
            continue
        if (limit is not None and len(texts) + len(log) >= limit) or (deadline is not None and time.time() > deadline):
            break
        try:
            r = s.get(f"{BASE}/ptr-pdfs/{year}/{doc}.pdf", headers=HEADERS, timeout=60)
        except requests.RequestException as exc:
            log.append({"doc_id": doc, "year": year, "status": f"error:{type(exc).__name__}", "n": 0})
            continue
        if r.status_code != 200:
            log.append({"doc_id": doc, "year": year, "status": f"http_{r.status_code}", "n": 0})
            continue
        time.sleep(pause)
        try:
            text = pdf_text(r.content)
        except Exception as exc:  # corrupt PDF
            log.append({"doc_id": doc, "year": year, "status": f"error:{type(exc).__name__}", "n": 0})
            continue
        if len(text.strip()) < 50:
            log.append({"doc_id": doc, "year": year, "status": "scan", "n": 0})
            continue
        texts.append({"doc_id": doc, "year": year, "first": f.get("First", ""), "last": f.get("Last", ""),
                      "filing_date": f["FilingDate"], "text": text})
    return pd.DataFrame(texts, columns=TEXT_COLUMNS), pd.DataFrame(log, columns=["doc_id", "year", "status", "n"])


def parse_store(texts: pd.DataFrame, legislators=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trade table rebuilt from the stored text. Returns (trades, log) with one log row per filing."""
    trades, log, ids = [], [], {}
    for f in texts.itertuples(index=False):
        try:
            rows = parse_ptr_text(f.text)
        except Exception as exc:
            log.append({"doc_id": f.doc_id, "year": f.year, "status": f"error:{type(exc).__name__}", "n": 0})
            continue
        key = (f.first, f.last, pd.Timestamp(f.filing_date).year)
        if legislators is not None and key not in ids:
            ids[key] = legislators.match(f.first, f.last, "house", pd.Timestamp(f.filing_date))
        for i, row in enumerate(rows):
            row.update({
                "trade_id": f"H-{f.doc_id}-{i + 1}", "chamber": "house", "member_id": ids.get(key),
                "member_name": f"{f.first} {f.last}".strip(), "disclosure_date": f.filing_date,
                "source_url": f"{BASE}/ptr-pdfs/{f.year}/{f.doc_id}.pdf",
            })
        trades.extend(rows)
        log.append({"doc_id": f.doc_id, "year": f.year, "status": "parsed" if rows else "no_rows", "n": len(rows)})
    out = finalize(pd.DataFrame(trades)) if trades else pd.DataFrame()
    return out, pd.DataFrame(log, columns=["doc_id", "year", "status", "n"])
