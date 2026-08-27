"""頑健性窓の開始日を DESIGN.md §10.1「開始日の規則（R1.1 で追加）」に従って計算し、
指数（__IDX__）のカバレッジを主評価窓と対称に確認する（診断専用。TASKS.md のタスクではない）。

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

指数カバレッジの確認（2026-08-26 実行条件の指示で主評価窓との対称比較に拡張）:
§6.1 の F 除外は欠損を通過扱いにするため、TOPIX(1306.T) が部分的に欠けていると
F10/F11/F12（d2_rs60/d2_rs120/d2_rsline_pos）が静かに無効化される。d2_rs60/rs120 は
T と T−60/T−120 の両方を参照するため、1日の欠落は最大で参照する2側に波及しうる。
優先すべきは補完の有無ではなく両窓の対称性（欠落率が同程度かどうか）である:
  - 両窓とも同程度 → 補完しない。NaN のまま欠損率を記録するだけでよい
  - 片方が明らかに多い → 両窓を前方補完（過去方向のみ・最大3営業日）した上で
    d2_rs60/d2_rs120/d2_rsline_pos の3列だけを再計算する（store の Close と IDX
    から再計算できるため、既存 replay の全面再生成は不要）。この判断はこの
    スクリプトの実測結果を見てから行う（結果を見て埋めるかどうかを決める）

1306.T は ETF のため配当（分配金）があり、権利落ち日に指数側だけ価格が一時的に
下振れする（docs/DATA_SOURCES.md 参照）。§6.1 はプール内百分位のため順位への
影響は乏しく、主評価窓も同条件のため比較は成立するが、健全性の記録として
「1306.T の単日下落が閾値を超えた日数」を窓ごとに1行だけ出す（修正はしない）。

backfill・MIN_HISTORY_BARS・MAX_EMPTY_DAY_RATE のいずれも変更しない（このスクリプトは
読み取りと計算のみ）。
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PREREGISTERED_FLOOR = pd.Timestamp("2017-03-01")
G3_LOOKBACK_BARS = 252  # DESIGN.md §2 G3: max(High[T-251..T])
POOL_WARMUP_DAYS = 20  # DESIGN.md §6.3 の直近20営業日プール
BUSINESS_DAYS_AFTER_EARLIEST = G3_LOOKBACK_BARS + POOL_WARMUP_DAYS  # 272
ROBUSTNESS_WINDOW_END = pd.Timestamp("2021-07-31")
# DESIGN.md §10.1（主評価窓。l1_report.md の対象期間表記と一致させる）
MAIN_WINDOW_START = pd.Timestamp("2021-08-01")
MAIN_WINDOW_END = pd.Timestamp("2026-01-30")
# 診断専用の閾値（スコアリングのパラメータではない）。1306.Tの配当落ち等による
# 「指数側だけの一時的な単日急落」を数えるための目安。DESIGN.md §12には追加しない
LARGE_DAILY_DECLINE_LOG_RETURN = -0.02


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


def report_index_coverage(label: str, idx_ohlcv: pd.DataFrame, calendar: pd.DatetimeIndex,
                          start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """指定期間の指数カバレッジ（全銘柄カレンダー基準の欠落率）と、1306.T の大幅単日
    下落の回数を報告する。欠落率（0〜1）を返す（比較用）。"""
    window_calendar = calendar[(calendar >= start) & (calendar <= end)]
    print(f"--- {label}: {start.date()} 〜 {end.date()} ---")
    if len(window_calendar) == 0:
        print("[NG] この期間の営業日がカレンダーに無い")
        return None
    idx_dates_in_window = set(idx_ohlcv.index) & set(window_calendar)
    missing = len(window_calendar) - len(idx_dates_in_window)
    missing_rate = missing / len(window_calendar)
    print(f"対象営業日数（全銘柄カレンダー基準）: {len(window_calendar)}")
    print(f"うち指数データがある日: {len(idx_dates_in_window)}")
    print(f"指数データが欠けている日: {missing} 日（{missing_rate:.2%}）")
    gaps = _find_gaps(window_calendar, idx_dates_in_window)
    if gaps:
        print(f"欠落区間: {len(gaps)} 件（連続日数が多い順に最大5件）")
        for g_start, g_end, g_len in sorted(gaps, key=lambda g: -g[2])[:5]:
            print(f"  {g_start.date()} 〜 {g_end.date()}（{g_len} 営業日）")
    else:
        print("欠落区間: なし")

    idx_in_window = idx_ohlcv[(idx_ohlcv.index >= start) & (idx_ohlcv.index <= end)]
    if len(idx_in_window) > 1:
        log_ret = np.log(idx_in_window["Close"]).diff()
        n_large_decline = int((log_ret < LARGE_DAILY_DECLINE_LOG_RETURN).sum())
        print(f"1306.T の単日下落が {LARGE_DAILY_DECLINE_LOG_RETURN:.0%} を超えた日"
              f"（診断のみ・配当落ち等の健全性記録、修正はしない）: {n_large_decline} 日")
    print()
    return missing_rate


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
        g3_start = min(g3_start, start_date)

    listed_path = cfg.reference_dir / "listed_latest.csv"
    if not listed_path.exists():
        print(f"[compute_robustness_start] {listed_path} が無いため、ユニバース通過数は計算できない")
        listed = None
    else:
        listed = normalize_listed(pd.read_csv(listed_path, dtype={"code": str}))

    print("=== 頑健性窓 開始日（DESIGN.md §10.1 開始日の規則） ===")
    print(f"store 最古日: {store_earliest.date()}")
    print(f"G3 計算可能日（{G3_LOOKBACK_BARS}本目。replay の実行開始日 --start）: {g3_start.date()}")
    print(f"store 最古日から {BUSINESS_DAYS_AFTER_EARLIEST} 営業日後: {calendar[pos].date()}")
    print(f"その翌営業日（規則の候補日）: {rule_candidate.date()}")
    print(f"事前登録の下限: {PREREGISTERED_FLOOR.date()}")
    print(f"→ 採用する集計対象開始日（stats_start。T-403の集計はここから）: {start_date.date()}")
    print("注記: replayの実行はg3_startから回すが、g3_start〜stats_start未満はwarmup=True。"
          "「対象営業日数」を数えるときは、集計対象(stats_start基準)とreplay実行対象"
          "(g3_start基準、warmupを含む)を区別して報告する（後述）")

    if listed is not None:
        universe_tickers = replay_universe_tickers(listed, ohlcv, start_date, MIN_HISTORY_BARS)
        print(f"集計対象開始日時点のユニバース通過銘柄数（現在の上場銘柄のうち履歴 "
              f"{MIN_HISTORY_BARS} 本以上。本番のuniverse.build＝流動性・株価等の filter とは別の、"
              f"replay専用の簡易な定義。DESIGN.md §10.1がそう定義している）: {len(universe_tickers)}")
    print("======================================================")
    print()

    if idx_ohlcv is None or len(idx_ohlcv) == 0:
        print("[NG] 指数データが無い。backfill/index の取得を確認すること")
        return 1

    print("=== 指数(__IDX__)カバレッジ: 主評価窓との対称比較 ===")
    main_rate = report_index_coverage("主評価窓", idx_ohlcv, calendar,
                                      MAIN_WINDOW_START, MAIN_WINDOW_END)
    print("[頑健性窓: replay実行対象(g3_start基準。warmup区間を含む、実際にreplayを回す範囲)]")
    robust_run_rate = report_index_coverage("頑健性窓(replay実行対象)", idx_ohlcv, calendar,
                                            g3_start, ROBUSTNESS_WINDOW_END)
    print("[頑健性窓: 集計対象(stats_start基準。T-403の集計に使う範囲。warmupを除く)]")
    robust_stats_rate = report_index_coverage("頑健性窓(集計対象)", idx_ohlcv, calendar,
                                              start_date, ROBUSTNESS_WINDOW_END)

    if main_rate is not None and robust_stats_rate is not None:
        diff = robust_stats_rate - main_rate
        print(f"欠落率の差（頑健性窓集計対象 − 主評価窓）: {diff:+.2%}")
        if abs(diff) < 0.02:
            print("→ 両窓とも同程度。前方補完は不要（NaNのまま欠損率を記録するだけでよい）")
        else:
            print("→ 片方が明らかに多い。両窓を前方補完（過去方向のみ・最大3営業日）した上で"
                  "d2_rs60/d2_rs120/d2_rsline_posを再計算するかどうか、設計責任者の判断を仰ぐこと"
                  "（このスクリプトはここでは補完を行わない）")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
