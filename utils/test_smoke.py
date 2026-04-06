from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.report import build_report
from utils.screener import run_screen


def make_df(seed: int, drift: float, last_boost: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 320
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, size=n)))
    base[-10:] *= (1 + np.linspace(0, last_boost, 10))
    close = pd.Series(base, index=dates)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.02, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.02, size=n))
    vol = pd.Series(rng.integers(800_000, 2_500_000, size=n), index=dates)
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Adj Close": close,
        "Volume": vol,
    }, index=dates)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    os.environ["LIQ_STRICT_ADV_MIN"] = "80000000"
    os.environ["LIQ_RELAX_ADV_MIN"] = "50000000"
    os.environ["LEADERS_MIN_RS_PCT"] = "75"
    os.environ["LEADERS_MAX_BB_RATIO"] = "1.20"
    universe = repo / "universe_jpx.csv"
    override = {
        "6501.T": make_df(1, 0.0020, 0.03),
        "6857.T": make_df(2, 0.0022, 0.04),
        "6920.T": make_df(3, 0.0021, 0.05),
        "7011.T": make_df(4, 0.0010, 0.01),
        "7203.T": make_df(5, -0.0003, 0.00),
        "7735.T": make_df(6, 0.0018, 0.02),
        "8035.T": make_df(7, 0.0023, 0.04),
        "8058.T": make_df(8, 0.0015, 0.01),
        "9984.T": make_df(9, 0.0002, 0.00),
    }
    screen = run_screen(
        today_str="2026-04-05",
        today_date=pd.Timestamp("2026-04-05").date(),
        mkt_score=68,
        delta3=5,
        macro_on=False,
        state={},
        universe_path=universe,
        ohlc_map_override=override,
    )
    trend = screen["trend_candidates"]
    leaders = screen["leader_candidates"]
    report = build_report(
        today_str="2026-04-05",
        market={"score": 68, "label": "strong", "lines": ["synthetic market"]},
        delta3=5,
        futures_chg=0.4,
        risk_on=True,
        macro_on=False,
        events_lines=[],
        no_trade=False,
        weekly_used=1,
        weekly_max=3,
        leverage=1.1,
        policy_lines=["smoke test"],
        trend_candidates=trend,
        leader_candidates=leaders,
        pos_text="保有なし",
        positions_df=pd.DataFrame(),
        out_dir=repo / "out",
    )
    assert Path(report.assets["summary_png"]).exists()
    assert Path(report.assets["trend_png"]).exists()
    assert Path(report.assets["leaders_png"]).exists()
    print({
        "trend": len(trend),
        "leaders": len(leaders),
        "assets": report.assets,
    })


if __name__ == "__main__":
    main()
