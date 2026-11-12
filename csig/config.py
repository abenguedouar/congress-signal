"""Configuration loader and shared constants."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

TRADE_COLUMNS = [
    "trade_id", "chamber", "member_id", "member_name", "owner", "ticker",
    "asset_name", "asset_type", "side", "trade_date", "disclosure_date",
    "amount_min", "amount_max", "source_url", "id_src", "note", "account",
]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mandate = ROOT / "csig" / "screen" / "mandate.yaml"      # the mandate's rules: present only where the screen is computed
    if mandate.exists():
        cfg.update(yaml.safe_load(mandate.read_text(encoding="utf-8")))
    return cfg
