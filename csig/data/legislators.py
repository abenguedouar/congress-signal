"""Members of Congress: identity, terms, party, leadership and committees.

Source: github.com/unitedstates/congress-legislators (public domain).
Committee membership is only published "as of today", so point-in-time
snapshots are rebuilt from the git history of that repository: the roles a
member held when a trade was disclosed, not the roles he holds now.
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

try:  # the C loader is ~10x faster on these large files
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _Loader

REPO_URL = "https://github.com/unitedstates/congress-legislators"


def ensure_repo(repo_dir, update: bool = False) -> Path:
    """Clone the public legislators repository (full history: committee snapshots) if it is not there."""
    import subprocess
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(repo_dir)], check=True)
    elif update:
        subprocess.run(["git", "-C", str(repo_dir), "pull", "-q"], check=False)
    return repo_dir


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", s.lower())


@dataclass
class Member:
    bioguide: str
    first: str
    last: str
    full: str
    nickname: str = ""
    terms: list = field(default_factory=list)          # dicts: type, start, end, party, state
    leadership: list = field(default_factory=list)     # dicts: title, chamber, start, end

    def term_at(self, date) -> dict | None:
        d = str(pd.Timestamp(date).date())
        for t in self.terms:
            if t["start"] <= d <= t["end"]:
                return t
        return None

    def chamber_at(self, date) -> str | None:
        t = self.term_at(date)
        return None if t is None else ("senate" if t["type"] == "sen" else "house")

    def party_at(self, date) -> str | None:
        t = self.term_at(date)
        return None if t is None else t.get("party")

    def seniority_years(self, date) -> float:
        first = min(t["start"] for t in self.terms)
        return max(0.0, (pd.Timestamp(date) - pd.Timestamp(first)).days / 365.25)

    def is_leader(self, date) -> bool:
        d = str(pd.Timestamp(date).date())
        return any(r["start"] <= d <= r.get("end", "9999-12-31") for r in self.leadership)


class Legislators:
    """Lookup table over current and historical members."""

    def __init__(self, repo_dir: str | Path, since_year: int = 2010):
        self.repo_dir = Path(repo_dir)
        self.members: dict[str, Member] = {}
        for name in ("legislators-current.yaml", "legislators-historical.yaml"):
            with open(self.repo_dir / name, encoding="utf-8") as fh:
                for rec in yaml.load(fh, Loader=_Loader):
                    terms = [t for t in rec["terms"] if t["end"] >= f"{since_year}-01-01"]
                    if not terms:
                        continue
                    n = rec["name"]
                    m = Member(
                        bioguide=rec["id"]["bioguide"],
                        first=n.get("first", ""), last=n.get("last", ""),
                        full=n.get("official_full") or f"{n.get('first', '')} {n.get('last', '')}",
                        nickname=n.get("nickname", ""),
                        terms=[{k: t.get(k) for k in ("type", "start", "end", "party", "state")} for t in terms],
                        leadership=rec.get("leadership_roles", []) or [],
                    )
                    self.members[m.bioguide] = m
        self._by_last: dict[str, list[Member]] = {}
        for m in self.members.values():
            self._by_last.setdefault(_norm(m.last), []).append(m)

    def get(self, bioguide: str) -> Member | None:
        return self.members.get(bioguide)

    def match(self, first: str, last: str, chamber: str | None = None, date=None) -> str | None:
        """Resolve a filer name to a bioguide id. Returns None when ambiguous."""
        last_n = _norm(last)
        # filings write "Smith Jr", "Van Hollen", "Capito, Shelley Moore"...
        cands = self._by_last.get(last_n) or [
            m for key, ms in self._by_last.items() for m in ms
            if key and (last_n.startswith(key) or last_n.endswith(key) or key.endswith(last_n))
        ]
        if date is not None and chamber:
            in_office = [m for m in cands if m.chamber_at(date) == chamber]
            cands = in_office or [m for m in cands if any(
                (t["type"] == "sen") == (chamber == "senate") for t in m.terms)]
        if len(cands) > 1:
            f = _norm(first.split()[0] if first.strip() else "")
            narrowed = [m for m in cands if f and (
                _norm(m.first).startswith(f[:3]) or _norm(m.nickname).startswith(f[:3])
                or f.startswith(_norm(m.first)[:3]))]
            cands = narrowed or cands
        return cands[0].bioguide if len(cands) == 1 else None

    # ------------------------------------------------------------------
    # Point-in-time committee membership, rebuilt from git history
    # ------------------------------------------------------------------
    def committee_snapshots(self, dates) -> dict[str, dict[str, set[str]]]:
        """For each date, {bioguide: {committee ids}} as published at that date."""
        out = {}
        for d in dates:
            snap = self._committees_asof(str(pd.Timestamp(d).date()))
            if len(snap) >= 100:          # the file is emptied at the start of each Congress: keep the previous one
                out[str(pd.Timestamp(d).date())] = snap
        return out

    @lru_cache(maxsize=128)
    def _committees_asof(self, date: str) -> dict[str, set[str]]:
        fname = "committee-membership-current.yaml"
        try:
            sha = subprocess.run(
                ["git", "-C", str(self.repo_dir), "log", "-1", "--format=%H", f"--before={date} 23:59", "--", fname],
                capture_output=True, text=True, check=True).stdout.strip()
            if sha:
                text = subprocess.run(["git", "-C", str(self.repo_dir), "show", f"{sha}:{fname}"],
                                      capture_output=True, text=True, check=True).stdout
            else:
                return {}
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}
        data = yaml.load(text, Loader=_Loader) or {}
        members: dict[str, set[str]] = {}
        for committee, rows in data.items():
            for r in rows or []:
                if r.get("bioguide"):
                    members.setdefault(r["bioguide"], set()).add(committee[:4])  # parent committee
                    title = (r.get("title") or "").lower()
                    if len(committee) == 4 and ("chair" in title or "ranking" in title):
                        members.setdefault("__leaders__", set()).add(r["bioguide"])  # chair / ranking member of a full committee
        return members


# Coarse link between a committee's jurisdiction and the SIC division of a company.
# Used for one feature only: "did the member sit on a committee overseeing this sector?"
COMMITTEE_SECTORS: dict[str, set[str]] = {
    "HSAS": {"defense"}, "SSAS": {"defense"},
    "HSIF": {"health", "energy", "tech", "telecom"}, "SSCM": {"tech", "telecom", "transport"},
    "SSEG": {"energy"}, "HSII": {"energy", "mining"},
    "HSBA": {"finance"}, "SSBK": {"finance"},
    "HSAG": {"food"}, "SSAF": {"food"},
    "SSHR": {"health"}, "HSWM": {"health", "finance"}, "SSFI": {"health", "finance"},
    "HSSY": {"tech", "defense"}, "HSPW": {"transport", "construction"}, "SSEV": {"transport", "construction", "energy"},
    "HSFA": {"defense"}, "SSFR": {"defense"}, "HLIG": {"defense", "tech"}, "SLIN": {"defense", "tech"},
}


def sic_sector(sic: str | int | None) -> str:
    """Map a 4-digit SIC code to the coarse sectors used above."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "other"
    if s in (3480, 3489, 3760, 3761, 3795, 3812) or 3720 <= s <= 3729:
        return "defense"
    if 1000 <= s <= 1299 or 1400 <= s <= 1499:
        return "mining"
    if 1300 <= s <= 1399 or 2900 <= s <= 2999 or 4900 <= s <= 4999:
        return "energy"
    if 1500 <= s <= 1799:
        return "construction"
    if 100 <= s <= 999 or 2000 <= s <= 2199:
        return "food"
    if 2830 <= s <= 2836 or 3840 <= s <= 3851 or 8000 <= s <= 8099:
        return "health"
    if 3570 <= s <= 3579 or 3660 <= s <= 3679 or 7370 <= s <= 7379:
        return "tech"
    if 4800 <= s <= 4899:
        return "telecom"
    if 4000 <= s <= 4799:
        return "transport"
    if 6000 <= s <= 6799:
        return "finance"
    return "other"


# Party holding the majority in each chamber, by Congress (start year of the two-year term).
# Used for one feature: "does the member's party control his chamber?"
# The entry for the Congress starting in January 2027 is filled in once it is seated.
MAJORITY = {
    2011: {"house": "Republican", "senate": "Democrat"},
    2013: {"house": "Republican", "senate": "Democrat"},
    2015: {"house": "Republican", "senate": "Republican"},
    2017: {"house": "Republican", "senate": "Republican"},
    2019: {"house": "Democrat", "senate": "Republican"},
    2021: {"house": "Democrat", "senate": "Democrat"},
    2023: {"house": "Republican", "senate": "Democrat"},
    2025: {"house": "Republican", "senate": "Republican"},
}


def majority_party(chamber: str, date) -> str | None:
    d = pd.Timestamp(date)
    start = d.year if d.year % 2 == 1 and (d.month, d.day) >= (1, 3) else (d.year - 1 if d.year % 2 == 0 else d.year - 2)
    return MAJORITY.get(start, {}).get(chamber)
