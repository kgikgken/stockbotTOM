"""store の履歴カバー範囲を出す（docs/BACKTEST.md §6）。

**窓の定義を黙って縮めないため**の事前確認。backfill は 2600 本なので、確認窓
（2017-03-15〜）の開始日まで届いていない可能性がある。届いていなければ、実際に
評価できる期間をそのまま報告する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockbot.config import Settings            # noqa: E402
from stockbot.data.store import OhlcvStore, from_long   # noqa: E402
from stockbot.validation.pattern_replay import CONFIRM_WINDOW, SEARCH_WINDOW  # noqa: E402


def main() -> int:
    cfg = Settings.from_env()
    ohlcv = from_long(OhlcvStore(cfg.store_dir, cfg.daily_dir).load())
    if not ohlcv:
        print("[coverage] store が空")
        return 0
    firsts = pd.Series({t: df.index.min() for t, df in ohlcv.items() if len(df)})
    lasts = pd.Series({t: df.index.max() for t, df in ohlcv.items() if len(df)})
    print(f"[coverage] 銘柄 {len(firsts)} / 全体 {firsts.min().date()}〜{lasts.max().date()}")
    for q in (0.05, 0.25, 0.5):
        print(f"[coverage] 開始日の{int(q * 100)}%点: {firsts.quantile(q).date()}")
    for name, (start, _end) in (("探索窓", SEARCH_WINDOW), ("確認窓", CONFIRM_WINDOW)):
        n = int((firsts <= start).sum())
        print(f"[coverage] {name}の開始日 {start.date()} までに履歴がある銘柄: "
              f"{n}/{len(firsts)}（{n / len(firsts):.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
