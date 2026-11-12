"""Descriptive statistics of a normalised trade table, and the herd / unusual split.

    python scripts/describe_trades.py data/house.parquet results/describe_house.json
"""
import json
import sys

import pandas as pd

from csig.data.trades import stock_trades
from csig.models.rpca import unusual_scores


def describe(trades: pd.DataFrame) -> dict:
    st = stock_trades(trades)
    buys = st[st["side"] == "buy"]
    per_member = buys["member_name"].value_counts()
    out = {
        "disclosure_from": str(st["disclosure_date"].min().date()),
        "disclosure_to": str(st["disclosure_date"].max().date()),
        "stock_trades": int(len(st)), "purchases": int(len(buys)), "sales": int((st["side"] == "sell").sum()),
        "members_trading": int(st["member_name"].nunique()), "members_buying": int(len(per_member)),
        "tickers": int(st["ticker"].nunique()),
        "lag_days_median": float(st["lag_days"].median()),
        "lag_days_p25_p75_p90": [float(x) for x in st["lag_days"].quantile([0.25, 0.75, 0.9])],
        "share_disclosed_after_45_days": float((st["lag_days"] > 45).mean()),
        "purchases_per_member_median": float(per_member.median()),
        "top5_members_share_of_purchases": float(per_member.head(5).sum() / len(buys)),
        "members_with_20_plus_purchases": int((per_member >= 20).sum()),
        "share_purchases_smallest_bucket": float((buys["amount_min"] <= 1001).mean()),
        "most_bought": buys["ticker"].value_counts().head(10).to_dict(),
    }
    scores, info = unusual_scores(st)
    herd = scores.sort_values("herd_part", ascending=False).groupby("ticker")["herd_part"].sum().nlargest(10)
    out["rpca"] = dict(info,
                       share_of_purchases_flagged_unusual=float((scores["unusual_share"] > 0.5).mean()),
                       herd_tickers=herd.round(1).to_dict(),
                       herd_overlap_with_most_bought=len(set(herd.index) & set(out["most_bought"])))
    return out


if __name__ == "__main__":
    res = describe(pd.read_parquet(sys.argv[1]))
    with open(sys.argv[2], "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
