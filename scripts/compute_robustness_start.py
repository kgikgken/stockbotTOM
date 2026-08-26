"""頑健性窓の開始日を DESIGN.md §10.1「開始日の規則（R1.1 で追加）」に従って計算する。

診断専用（TASKS.md のタスクではない）。store（backfill 済み）を読み、

    頑健性窓の開始日 = 「store 最古日から 272 営業日後の翌営業日」と「2017-03-01」の
    いずれか遅い方

を実測データで計算し、算出した開始日時点のユニバース通過銘柄数
（replay.py の定義: 現在の上場銘柄のうち T 時点で履歴 250 本以上ある銘柄）とあわせて出力する。

「営業日」は実際の取引日カレンダー（store 内の全銘柄・指数の日付の和集合）で数える。
pandas の bdate_range（土日のみ除外）は日本の祝日を含んでしまい、272 営業日の実質的な
バーの本数（G3 の 252 本＋§6.3 プールの 20 日）と整合しないため使わない。

backfill・MIN_HISTORY_BARS・MAX_EMPTY_DAY_RATE のいずれも変更しない（このスクリプトは
読み取りと計算のみ）。
"""
from __future__ import annotations

import sys

import pandas as pd

PREREGISTERED_FLOOR = pd.Timestamp("2017-03-01")
BUSINESS_DAYS_AFTER_EARLIEST = 272  # G3 の 252 本 + §6.3 プールの 20 日ウォームアップ


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

    pos = BUSINESS_DAYS_AFTER_EARLIEST  # 0=最古日そのもの。272営業日後 = calendar[272]
    next_pos = pos + 1  # その翌営業日
    if next_pos >= len(calendar):
        print(f"[compute_robustness_start] カレンダーが {next_pos} 営業日に届かない"
              f"（全 {len(calendar)} 日）。backfill を確認すること")
        return 1
    rule_candidate = calendar[next_pos]

    start_date = max(rule_candidate, PREREGISTERED_FLOOR)
    # 2017-03-01 が採用される場合、カレンダー上の実際の取引日に丸める（同日以降の最初の取引日）
    if start_date == PREREGISTERED_FLOOR and PREREGISTERED_FLOOR not in calendar:
        later = calendar[calendar >= PREREGISTERED_FLOOR]
        if len(later) == 0:
            print("[compute_robustness_start] 2017-03-01 以降の取引日がカレンダーに無い")
            return 1
        start_date = later[0]

    listed_path = cfg.reference_dir / "listed_latest.csv"
    if not listed_path.exists():
        print(f"[compute_robustness_start] {listed_path} が無いため、ユニバース通過数は計算できない")
        listed = None
    else:
        listed = normalize_listed(pd.read_csv(listed_path, dtype={"code": str}))

    print("=== 頑健性窓 開始日（DESIGN.md §10.1 開始日の規則） ===")
    print(f"store 最古日: {store_earliest.date()}")
    print(f"store 最古日から {BUSINESS_DAYS_AFTER_EARLIEST} 営業日後: {calendar[pos].date()}")
    print(f"その翌営業日（規則の候補日）: {rule_candidate.date()}")
    print(f"事前登録の下限: {PREREGISTERED_FLOOR.date()}")
    print(f"→ 採用する開始日（いずれか遅い方）: {start_date.date()}")

    if listed is not None:
        universe_tickers = replay_universe_tickers(listed, ohlcv, start_date, MIN_HISTORY_BARS)
        print(f"開始日時点のユニバース通過銘柄数（履歴 {MIN_HISTORY_BARS} 本以上）: "
              f"{len(universe_tickers)}")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
