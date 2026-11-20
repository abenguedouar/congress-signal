"""Writes the snapshot of the test account that the public page reads while the market is open.

    python3 scripts/live_snapshot.py OUT.json [--compare URL] [--launch YYYY-MM-DD]

Standard library only. With --compare, tells whether the page would look different from the snapshot already
published at URL (so that nothing is pushed when the market is closed and nothing moved).
--launch pretends the portfolio started on another day: a test of the broker's answers, never to be published.
"""
import importlib.util
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("snapshot", ROOT / "csig" / "live" / "snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def main(out: str, compare: str | None = None, pretend: str | None = None) -> None:
    live, context = ROOT / "data" / "store" / "live", {}
    for name, key in (("context.json", None), ("closes.json", "closes"), ("reports.json", "reports"), ("baseline.json", "baseline")):
        if (live / name).exists():
            data = json.loads((live / name).read_text())
            context.update(data if key is None else {key: data})
    settings = (ROOT / "config.yaml").read_text()
    launch = pretend or re.search(r"launch_date:\s*(\d{4}-\d{2}-\d{2})", settings).group(1)
    blob = snapshot.build(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], launch, context)
    changed = True
    if compare:
        try:
            with urllib.request.urlopen(compare, timeout=20) as r:
                changed = not snapshot.same_picture(blob, json.loads(r.read().decode()))
        except Exception:
            changed = True
    Path(out).write_text(json.dumps(blob, separators=(",", ":")))
    series = blob["series"]
    print(json.dumps({"mode": blob["mode"], "market_open": blob["market"].get("open"), "equity": blob["equity"], "cash": blob["cash"],
                      "positions": len(blob["positions"]), "days": len(series["dates"]), "last_day": series["dates"][-1:],
                      "portfolio": series["p"][-1:], "benchmark": series["s"][-1:], "orders": len(blob["activity"]), "changed": changed,
                      "smallest": [(p["ticker"], p["qty"], round(p["value"], 2)) for p in blob["positions"][-3:]]}))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\nmode={'test' if pretend else blob['mode']}\n")
    if pretend:
        print(json.dumps({"series": series, "market": blob["market"], "day": blob["day"], "first_positions": blob["positions"][:2], "first_orders": blob["activity"][:2]}))


if __name__ == "__main__":
    args = sys.argv[1:]
    option = lambda name: args[args.index(name) + 1] if name in args else None
    main(args[0], option("--compare"), option("--launch"))
