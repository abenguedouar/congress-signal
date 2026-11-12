"""Fama-French daily factors (five factors plus momentum), from Kenneth French's data library."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "mom": "F-F_Momentum_Factor_daily_CSV.zip"}


def _parse(text: str) -> pd.DataFrame:
    """The files carry free-text headers and footers; keep the rows that start with a yyyymmdd date."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",") or ln.strip().lower().startswith("date"))
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for ln in lines[start + 1:]:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == len(header) and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0              # published in percent


def load_factors(cache_dir: str | Path, refresh: bool = False) -> pd.DataFrame:
    path = Path(cache_dir) / "factors.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frames = []
    for name in FILES.values():
        r = requests.get(f"{BASE}/{name}", timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            frames.append(_parse(z.read(z.namelist()[0]).decode("latin-1")))
    out = pd.concat(frames, axis=1).dropna()
    out.columns = [c.replace("Mkt-RF", "MKT") for c in out.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out
