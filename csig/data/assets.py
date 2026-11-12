"""Reading the asset cell of a House filing: owner, ticker and asset type.

Seen on the full history (2014-2026):

* the text layer of many PDFs has broken letter case ("(aaPL) [sT]"), so
  nothing here may depend on case;
* before April 2018 the form had no asset-type code: the type is then read
  from the wording (a cell that ends with a ticker and is not an option or a
  fund is a stock);
* the cell may start with a transaction id or an owner code, and may carry a
  holding path ("Family Trust > Apple Inc.").

``id_src`` records how each row was identified, so that every result can be
re-computed on the rows that needed no inference:

    coded    asset-type code printed in the filing
    ticker   no type code; ticker printed; type read from the wording
    ""       not identified (kept as "other")
"""
from __future__ import annotations

import re

OWNER = {"SP": "spouse", "JT": "joint", "DC": "child", "DE": "child"}
_PREFIX = re.compile(r"^(?:\d{7,}\s+)?(?:((?i:SP|JT|DC)|DE)\s+)?")
_CODE = re.compile(r"\[\s*([A-Za-z0-9]{1,3})\s*\]")
_PAREN_TICKER = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9]{0,5}(?:[.\-/][A-Za-z])?)\s*\)")
_NOT_TICKER = {"ADR", "ADS", "REIT", "NEW", "THE", "INC", "ETF", "LLC", "LP", "PLC", "NA", "MLP"}
_MONTH = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
_OPTION_HEAD = re.compile(rf"^([A-Za-z]{{1,5}})\s+{_MONTH}\w*\s+\d{{1,2}}\s+(?:\d{{2,4}}\s+)?\$?[\d.]+\s+(?:CALL|PUT)", re.I)
_OPTION = re.compile(r"\b(?:CALLS?|PUTS?|OPTIONS?)\b", re.I)
_NOT_STOCK = re.compile(
    r"\d\s*%|\b(?:ETF|ETN|FUND|FUNDS|INDEX|ISHARES|SPDR|VANGUARD|PROSHARES|POWERSHARES|DIREXION|NTS|B/E|PFD)\b", re.I)


def clean_symbol(raw: str | None) -> str | None:
    if not raw:
        return None
    t = str(raw).strip().upper().replace("/", ".").replace("-", ".")
    if t in _NOT_TICKER or not re.match(r"^[A-Z][A-Z0-9]{0,4}(\.[A-Z])?$", t):
        return None
    return t


def interpret(raw: str) -> dict:
    """Split one asset cell into owner, ticker, name, asset type and ``id_src``."""
    s = re.sub(r"\s+", " ", raw or "").strip()
    s = re.split(r"\s(?:>|=>|⇒|→)\s", s)[-1]            # keep the asset, drop the holding path
    owner = None
    m = _PREFIX.match(s)
    if m and m.end():
        owner = OWNER.get((m.group(1) or "").upper())
        s = s[m.end():]
    code_m = _CODE.search(s)
    code = code_m.group(1).upper() if code_m else None
    head = s[:code_m.start()] if code_m else s
    found = [x for x in (clean_symbol(x) for x in _PAREN_TICKER.findall(head)) if x]
    ticker = found[-1] if found else None
    name = re.sub(r"\s*\(\s*[A-Za-z][A-Za-z0-9.\-/]{0,7}\s*\)\s*$", "", head).strip(" -")

    if code is not None:
        atype = "stock" if code == "ST" else ("option" if code == "OP" else "other")
        src = "coded"
    else:                                               # form without a type code: read the wording
        opt = _OPTION_HEAD.match(name)
        if opt:
            atype, ticker, src = "option", clean_symbol(opt.group(1)), "ticker"
        elif _OPTION.search(name):
            atype, src = "option", ("ticker" if ticker else "")
        elif ticker and not _NOT_STOCK.search(name):
            atype, src = "stock", "ticker"
        else:
            atype, src = "other", ""
    return {"owner": owner, "ticker": ticker, "asset_name": name, "asset_type": atype, "id_src": src}
