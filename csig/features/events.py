"""Event table: one row per disclosed purchase, with features and outcomes.

Timing convention (the whole project rests on it):

    trade_date  ->  disclosure_date  ->  t0  ->  t0 + H
                    (filing public)     first trading day strictly after disclosure

Features use information available at t0 at the latest. The outcome is the
stock's return from the close of t0 to the close of t0 + H trading days,
minus the benchmark's return over the same days ("abnormal return").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.legislators import COMMITTEE_SECTORS, majority_party, sic_sector


def _pos_after(calendar: pd.DatetimeIndex, dates: pd.Series, strict: bool) -> np.ndarray:
    side = "right" if strict else "left"
    return np.searchsorted(calendar.values, dates.values.astype("datetime64[ns]"), side=side)


def build_events(trades: pd.DataFrame, prices: pd.DataFrame, benchmark: str, horizons: list[int],
                 legislators=None, committees: dict | None = None, sic_by_ticker: dict | None = None,
                 winsor: float = 0.01, state_by_ticker: dict | None = None) -> pd.DataFrame:
    """``trades`` comes from ``data.trades.stock_trades``; ``prices`` is the adjusted wide frame."""
    cal = prices.index
    ev = trades[trades["ticker"].isin(prices.columns)].copy().reset_index(drop=True)
    i0 = _pos_after(cal, ev["disclosure_date"], strict=True)
    itrade = _pos_after(cal, ev["trade_date"], strict=False)
    ok = (i0 < len(cal)) & (itrade < len(cal))
    ev, i0, itrade = ev[ok].reset_index(drop=True), i0[ok], itrade[ok]
    ev["t0"] = cal[i0]

    px = prices.to_numpy(dtype=float)
    col = prices.columns.get_indexer(ev["ticker"])
    b = prices.columns.get_loc(benchmark)
    p0, b0 = px[i0, col], px[i0, b]
    with np.errstate(divide="ignore", invalid="ignore"):
        ev["runup"] = (p0 / px[itrade, col] - 1) - (b0 / px[itrade, b] - 1)
        for h in horizons:
            i1 = np.minimum(i0 + h, len(cal) - 1)
            ab = (px[i1, col] / p0 - 1) - (px[i1, b] / b0 - 1)
            ab[i0 + h > len(cal) - 1] = np.nan          # outcome not yet observed
            ev[f"ab_{h}"] = ab
            ev[f"end_{h}"] = cal[i1]
    for h in horizons:
        lo, hi = ev[f"ab_{h}"].quantile([winsor, 1 - winsor])
        ev[f"ab_{h}"] = ev[f"ab_{h}"].clip(lo, hi)

    # --- features known at disclosure --------------------------------------
    ev["log_amount"] = np.log10(ev["amount_mid"].clip(lower=1000))
    ev["senate"] = ev["chamber"].eq("senate").astype(float)
    for o in ("spouse", "joint", "child"):
        ev[f"owner_{o}"] = ev["owner"].eq(o).astype(float)
    ev["filing_size"] = ev.groupby("source_url")["trade_id"].transform("count").astype(float)
    ev = ev.sort_values(["disclosure_date", "trade_id"]).reset_index(drop=True)
    ev["member_key"] = ev["member_id"].fillna(ev["member_name"])
    ev["member_n_prior"] = ev.groupby("member_key").cumcount().astype(float)
    ev["first_buy"] = (ev.groupby(["member_key", "ticker"]).cumcount() == 0).astype(float)
    ev["new_trader"] = (ev["member_n_prior"] < 3).astype(float)       # hardly any disclosed purchase before this one
    ev["consensus_30d"] = _consensus(ev, 30)

    for c in ("via_option", "managed"):
        if c not in ev:
            ev[c] = 0.0
    ev["republican"], ev["seniority"], ev["leader"], ev["committee_match"], ev["freshman"] = 0.0, 0.0, 0.0, 0.0, 0.0
    ev["committee_leader"], ev["majority"], ev["home_state"] = 0.0, 0.0, 0.0
    if legislators is not None:
        snap_dates = sorted(committees) if committees else []
        for idx, r in ev.iterrows():
            m = legislators.get(r["member_id"]) if r["member_id"] else None
            if m is None:
                continue
            d = r["disclosure_date"]
            ev.at[idx, "republican"] = float(m.party_at(d) == "Republican")
            ev.at[idx, "seniority"] = m.seniority_years(d)
            ev.at[idx, "freshman"] = float(m.seniority_years(d) < 2.0)   # first term in Congress
            ev.at[idx, "leader"] = float(m.is_leader(d))
            term = m.term_at(d)
            if term is not None:
                ev.at[idx, "majority"] = float(m.party_at(d) == majority_party(m.chamber_at(d), d))
                if state_by_ticker:
                    ev.at[idx, "home_state"] = float(state_by_ticker.get(r["ticker"]) == term.get("state"))
            if snap_dates:
                past = [s for s in snap_dates if s <= str(d.date())]
                if past:
                    ev.at[idx, "committee_leader"] = float(m.bioguide in committees[past[-1]].get("__leaders__", set()))
            if snap_dates and sic_by_ticker:
                past = [s for s in snap_dates if s <= str(d.date())]
                if past:
                    sectors = set().union(*[COMMITTEE_SECTORS.get(c, set())
                                            for c in committees[past[-1]].get(m.bioguide, set())])
                    ev.at[idx, "committee_match"] = float(sic_sector(sic_by_ticker.get(r["ticker"])) in sectors)
    return ev


def _consensus(ev: pd.DataFrame, days: int) -> pd.Series:
    """Number of other members who disclosed a purchase of the same ticker in the previous ``days``."""
    out = np.zeros(len(ev))
    for _, g in ev.groupby("ticker"):
        d = g["disclosure_date"].values.astype("datetime64[D]")
        who = g["member_key"].values
        for k in range(len(g)):
            window = (d <= d[k]) & (d >= d[k] - np.timedelta64(days, 'D'))
            out[g.index[k]] = len(set(who[window]) - {who[k]})
    return pd.Series(out, index=ev.index)


FEATURES = ["lag_days", "runup", "log_amount", "senate", "owner_spouse", "owner_joint", "owner_child",
            "filing_size", "member_n_prior", "first_buy", "new_trader", "consensus_30d",
            "republican", "seniority", "freshman", "leader", "committee_match", "committee_leader",
            "majority", "home_state", "via_option", "managed"]
