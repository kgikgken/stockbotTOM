"""Fetch raw inputs from yfinance and save them as-is (docs/theme/README.md "Definitions").

Only data on or before asof is requested for daily bars. Nothing after asof
(subsequent price) is fetched.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, datetime, timedelta, timezone

from .config import RAW_DIR, UNIVERSE_DIR

INFO_KEYS: tuple[str, ...] = ("longName", "shortName", "marketCap", "currency", "quoteType")
INCOME_ITEMS: tuple[str, ...] = ("Operating Income", "Total Operating Income As Reported")
HISTORY_CALENDAR_DAYS: int = 200  # enough calendar days for 60 sessions


def _num(x: object) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def fetch_one(code: str, asof: date) -> tuple[dict, list[dict]]:
    """Fetch info, annual income items and daily bars for one code."""
    import yfinance as yf  # lazy: tests do not need it

    t = yf.Ticker(f"{code}.T")
    rec: dict = {"code": code, "symbol": f"{code}.T", "errors": {}}
    try:
        info = t.info or {}
        rec["info"] = {k: info.get(k) for k in INFO_KEYS}
    except Exception as e:  # recorded, not raised: computability is what we measure
        rec["info"] = None
        rec["errors"]["info"] = f"{type(e).__name__}: {e}"
    try:
        inc = t.income_stmt
        items: dict[str, dict[str, float | None]] = {}
        if inc is not None and not inc.empty:
            for item in INCOME_ITEMS:
                if item in inc.index:
                    items[item] = {pd_ts.strftime("%Y-%m-%d"): _num(v)
                                   for pd_ts, v in inc.loc[item].items()}
        rec["income_annual"] = items
    except Exception as e:
        rec["income_annual"] = None
        rec["errors"]["income_annual"] = f"{type(e).__name__}: {e}"
    bars: list[dict] = []
    try:
        h = t.history(start=(asof - timedelta(days=HISTORY_CALENDAR_DAYS)).isoformat(),
                      end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False)
        for ts, row in h.iterrows():
            d = ts.strftime("%Y-%m-%d")
            if d <= asof.isoformat():
                bars.append({"code": code, "date": d, "close": _num(row.get("Close")),
                             "volume": _num(row.get("Volume"))})
    except Exception as e:
        rec["errors"]["history"] = f"{type(e).__name__}: {e}"
    return rec, bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="last completed session, YYYY-MM-DD")
    ap.add_argument("--universe", default=str(UNIVERSE_DIR / "semicon_frontend.csv"))
    args = ap.parse_args()
    asof = date.fromisoformat(args.asof)
    with open(args.universe, encoding="utf-8") as f:
        codes = [r["code"] for r in csv.DictReader(f)]
    out = RAW_DIR / asof.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    all_bars: list[dict] = []
    for i, code in enumerate(codes, 1):
        rec, bars = fetch_one(code, asof)
        recs.append(rec)
        all_bars.extend(bars)
        print(f"[{i}/{len(codes)}] {code} bars={len(bars)} errors={list(rec['errors'])}")
        time.sleep(0.5)
    meta = {"asof": asof.isoformat(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "yfinance", "records": recs}
    (out / "info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "date", "close", "volume"])
        w.writeheader()
        w.writerows(all_bars)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
