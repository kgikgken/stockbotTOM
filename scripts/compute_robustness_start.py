"""頑健性窓の開始日を DESIGN.md §10.1「開始日の規則（R1.1 で追加）」に従って計算し、
指数（__IDX__）のカバレッジを確認する（診断専用。TASKS.md のタスクではない）。

store（backfill 済み）を読み、

    頑健性窓の開始日 = 「store 最古日から 272 営業日後の翌営業日」と「2017-03-01」の
    いずれか遅い方

を実測データで計算する。272 = G3 の 252 本ルックバック + §6.3 プールの 20 日
ウォームアップなので、G3 が計算可能になる最初の日（252 本目、= 272 営業日規則の
基準日から 20 営業日前）も併せて報告する。実際の replay は G3 計算可能日から回し、
開始日未満の行は warmup=True として集計から除外する（run_replay の stats_start）。

「営業日」は実際の取引日カレンダー（store 内の全銘柄・指数の日付の和集合）で数える。
pandas の bdate_range（土日のみ除外）は日本の祝日を含んでしまい、272 営業日の実質的な
バーの本数と整合しないため使わない。

指数カバレッジの確認: §6.1 の F 除外は欠損を通過扱いにするため、TOPIX(1306.T) が
部分的に欠けていると F10/F11/F12（d2_rs60/d2_rs120/d2_rsline_pos）が静かに無効化され、
主評価窓とは別のフィルタを測ることになる（今回最も危険な失敗モード）。頑健性窓の
対象期間全体で指数データの欠落が無いかを検査する。

backfill・MIN_HISTORY_BARS・MAX_EMPTY_DAY_RATE のいずれも変更しない（このスクリプトは
読み取りと計算のみ）。
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import pandas as pd

PREREGISTERED_FLOOR = pd.Timestamp("2017-03-01")
G3_LOOKBACK_BARS = 252  # DESIGN.md §2 G3: max(High[T-251..T])
POOL_WARMUP_DAYS = 20  # DESIGN.md §6.3 の直近20営業日プール
BUSINESS_DAYS_AFTER_EARLIEST = G3_LOOKBACK_BARS + POOL_WARMUP_DAYS  # 272
ROBUSTNESS_WINDOW_END = pd.Timestamp("2021-07-31")


def _find_gaps(calendar: pd.DatetimeIndex, present: set) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """calendar のうち present に無い日を、連続区間（開始日, 終了日, 日数）でまとめる。"""
    gaps = []
    run_start = None
    run_len = 0
    prev = None
    for d in calendar:
        if d not in present:
            if run_start is None:
                run_start = d
            run_len += 1
            prev = d
        else:
            if run_start is not None:
                gaps.append((run_start, prev, run_len))
            run_start = None
            run_len = 0
    if run_start is not None:
        gaps.append((run_start, prev, run_len))
    return gaps


def main() -> int:
    from stockbot.config import Settings
    from stockbot.data.jpx_lists import normalize_listed
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long
    from stockbot.validation.replay import MIN_HISTORY_BARS, replay_universe_tickers

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    long_df = store.load()
    if long_df is None or len(long_df) == 0:
        print("[compute_robustness_start] store が空。先に backfill を実行すること")
        return 1
    ohlcv = from_long(long_df)
    idx_ohlcv = ohlcv.pop(IDX_TICKER, None)

    all_dates = set()
    for df in ohlcv.values():
        if df is not None and len(df):
            all_dates.update(df.index)
    if idx_ohlcv is not None and len(idx_ohlcv):
        all_dates.update(idx_ohlcv.index)
    if not all_dates:
        print("[compute_robustness_start] 取引日カレンダーが空")
        return 1

    calendar = pd.DatetimeIndex(sorted(all_dates))
    store_earliest = calendar[0]

    g3_pos = G3_LOOKBACK_BARS - 1  # 0=最古日そのもの。252本目 = calendar[251]
    pos = BUSINESS_DAYS_AFTER_EARLIEST  # 272営業日後 = calendar[272]
    next_pos = pos + 1  # その翌営業日
    if next_pos >= len(calendar):
        print(f"[compute_robustness_start] カレンダーが {next_pos} 営業日に届かない"
              f"（全 {len(calendar)} 日）。backfill を確認すること")
        return 1
    g3_start = calendar[g3_pos]
    rule_candidate = calendar[next_pos]

    start_date = max(rule_candidate, PREREGISTERED_FLOOR)
    if start_date == PREREGISTERED_FLOOR and PREREGISTERED_FLOOR not in calendar:
        later = calendar[calendar >= PREREGISTERED_FLOOR]
        if len(later) == 0:
            print("[compute_robustness_start] 2017-03-01 以降の取引日がカレンダーに無い")
            return 1
        start_date = later[0]
        # 下限側が採用される場合、replay の実開始日(g3_start)はこの下限に対応する
        # G3計算可能日でなければならない。272営業日規則が下限を上回るのが通常ケースなので
        # ここに来るのは事前登録の下限そのものが272営業日規則より遅い、想定外の store 状態
        g3_start = min(g3_start, start_date)

    listed_path = cfg.reference_dir / "listed_latest.csv"
    if not listed_path.exists():
        print(f"[compute_robustness_start] {listed_path} が無いため、ユニバース通過数は計算できない")
        listed = None
    else:
        listed = normalize_listed(pd.read_csv(listed_path, dtype={"code": str}))

    print("=== 頑健性窓 開始日（DESIGN.md §10.1 開始日の規則） ===")
    print(f"store 最古日: {store_earliest.date()}")
    print(f"G3 計算可能日（{G3_LOOKBACK_BARS}本目。replay の実開始日）: {g3_start.date()}")
    print(f"store 最古日から {BUSINESS_DAYS_AFTER_EARLIEST} 営業日後: {calendar[pos].date()}")
    print(f"その翌営業日（規則の候補日）: {rule_candidate.date()}")
    print(f"事前登録の下限: {PREREGISTERED_FLOOR.date()}")
    print(f"→ 採用する集計対象開始日(stats_start): {start_date.date()}")
    print(f"→ replay の実行開始日(--start、warmup区間の先頭): {g3_start.date()}")

    if listed is not None:
        universe_tickers = replay_universe_tickers(listed, ohlcv, start_date, MIN_HISTORY_BARS)
        print(f"集計対象開始日時点のユニバース通過銘柄数（履歴 {MIN_HISTORY_BARS} 本以上）: "
              f"{len(universe_tickers)}")
    print("======================================================")

    print()
    print("=== 指数(__IDX__)カバレッジ確認（頑健性窓: "
          f"{g3_start.date()} 〜 {ROBUSTNESS_WINDOW_END.date()}） ===")
    if idx_ohlcv is None or len(idx_ohlcv) == 0:
        print("[NG] 指数データが無い。backfill/index の取得を確認すること")
        return 1
    window_calendar = calendar[(calendar >= g3_start) & (calendar <= ROBUSTNESS_WINDOW_END)]
    idx_dates_in_window = set(idx_ohlcv.index) & set(window_calendar)
    print(f"指数データの全期間: {idx_ohlcv.index.min().date()} 〜 {idx_ohlcv.index.max().date()}"
          f"（{len(idx_ohlcv)} 本）")
    print(f"頑健性窓の対象営業日数（全銘柄カレンダー基準）: {len(window_calendar)}")
    print(f"うち指数データがある日: {len(idx_dates_in_window)}")
    missing = len(window_calendar) - len(idx_dates_in_window)
    missing_rate = missing / len(window_calendar) if len(window_calendar) else 0.0
    print(f"指数データが欠けている日: {missing} 日（{missing_rate:.2%}）")
    gaps = _find_gaps(window_calendar, idx_dates_in_window)
    if gaps:
        print(f"欠落区間: {len(gaps)} 件（連続日数が多い順に最大10件）")
        for g_start, g_end, g_len in sorted(gaps, key=lambda g: -g[2])[:10]:
            print(f"  {g_start.date()} 〜 {g_end.date()}（{g_len} 営業日）")
    else:
        print("欠落区間: なし")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
