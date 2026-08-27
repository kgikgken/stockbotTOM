"""compute_robustness_start.py が「指数データが欠けている日」として検出した日付が、
実際に取引所が開いていた営業日か、それとも取引所休場日（カレンダー由来の誤検出）かを
確認する（診断専用。TASKS.md のタスクではない）。

compute_robustness_start.py の「calendar」は store 内の全銘柄・指数の日付の和集合
であり、どれか1銘柄でも特定の日付のデータを持っていれば「営業日」として数える。
ある銘柄のデータに、取引所が閉まっている日（元日等）の異常な1件が混入していると、
その日が「営業日」として数えられ、指数側にその日のデータが無いことが「指数の欠落」
として誤検出される。7203.T（トヨタ）を基準に、各候補日について実際に何銘柄が
データを持っているかを数える。既定では compute_robustness_start.py と同じロジックで
頑健性窓（集計対象）の欠落日を自動検出して全件チェックする（--dates で個別指定も可）。

store を変更しない（読み取りのみ）。
"""
from __future__ import annotations

import sys

import pandas as pd

REFERENCE_TICKER = "7203.T"


def _default_gap_dates() -> list:
    """compute_robustness_start.py と同じロジックで頑健性窓(集計対象)の欠落日を検出する。"""
    from compute_robustness_start import (
        BUSINESS_DAYS_AFTER_EARLIEST, G3_LOOKBACK_BARS, PREREGISTERED_FLOOR,
        ROBUSTNESS_WINDOW_END, _find_gaps,
    )
    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_ohlcv = ohlcv.pop(IDX_TICKER, None)

    all_dates = set()
    for df in ohlcv.values():
        if df is not None and len(df):
            all_dates.update(df.index)
    if idx_ohlcv is not None and len(idx_ohlcv):
        all_dates.update(idx_ohlcv.index)
    calendar = pd.DatetimeIndex(sorted(all_dates))

    pos = BUSINESS_DAYS_AFTER_EARLIEST
    next_pos = pos + 1
    g3_start = calendar[G3_LOOKBACK_BARS - 1]
    rule_candidate = calendar[next_pos]
    start_date = max(rule_candidate, PREREGISTERED_FLOOR)
    if start_date == PREREGISTERED_FLOOR and PREREGISTERED_FLOOR not in calendar:
        later = calendar[calendar >= PREREGISTERED_FLOOR]
        start_date = later[0]
        g3_start = min(g3_start, start_date)

    window_calendar = calendar[(calendar >= start_date) & (calendar <= ROBUSTNESS_WINDOW_END)]
    idx_dates_in_window = set(idx_ohlcv.index) & set(window_calendar)
    gaps = _find_gaps(window_calendar, idx_dates_in_window)
    out = []
    for g_start, g_end, _g_len in gaps:
        out.extend(pd.bdate_range(g_start, g_end))
    return sorted(out)


def main() -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long

    ap = argparse.ArgumentParser(prog="python scripts/check_index_gap_dates.py")
    ap.add_argument("--dates", nargs="+", default=None,
                    help="YYYY-MM-DD (複数可)。省略時は頑健性窓の欠落日を自動検出する")
    args = ap.parse_args()

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_ohlcv = ohlcv.pop(IDX_TICKER, None)

    dates = [pd.Timestamp(d) for d in args.dates] if args.dates else _default_gap_dates()
    print(f"検証対象: {len(dates)} 日")

    reference = ohlcv.get(REFERENCE_TICKER)
    if reference is None:
        print(f"[WARN] {REFERENCE_TICKER} が store に無い")

    total_tickers = sum(1 for df in ohlcv.values() if df is not None and len(df))
    n_holiday_like = n_real_gap_like = n_ambiguous = 0

    print(f"=== 候補日の検証（基準銘柄: {REFERENCE_TICKER}） ===")
    for date_t in dates:
        n_tickers_with_date = sum(
            1 for df in ohlcv.values() if df is not None and date_t in df.index)
        idx_has_date = idx_ohlcv is not None and date_t in idx_ohlcv.index
        ref_has_date = reference is not None and date_t in reference.index
        frac = n_tickers_with_date / total_tickers if total_tickers else 0.0
        print(f"--- {date_t.date()} ({date_t.day_name()}) ---")
        print(f"  {REFERENCE_TICKER} にこの日のデータあり: {ref_has_date}")
        print(f"  指数(__IDX__)にこの日のデータあり: {idx_has_date}")
        print(f"  この日のデータを持つ銘柄数: {n_tickers_with_date} / {total_tickers} ({frac:.2%})")
        if n_tickers_with_date <= max(2, total_tickers * 0.01):
            print("  → ほぼ全銘柄がこの日を持たない。取引所休場日（カレンダー由来の誤検出）の可能性が高い")
            n_holiday_like += 1
        elif frac < 0.5:
            print("  → 一部銘柄のみがこの日を持つ。データ異常銘柄が混入している可能性")
            n_ambiguous += 1
        else:
            print("  → 大多数の銘柄がこの日を持つ。実際の営業日で、指数側だけの欠落である可能性が高い")
            n_real_gap_like += 1
        print()

    print(f"内訳: 休場日らしき日 {n_holiday_like} / 実際の欠落らしき日 {n_real_gap_like} / "
          f"判定困難 {n_ambiguous}（全 {len(dates)} 日）")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
