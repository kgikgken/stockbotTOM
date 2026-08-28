"""adjust.check_splits を全銘柄・store の完全マージ済み系列（単発fetchの窓に
限らない）に対して実行する（診断専用。TASKS.mdのタスクではない）。

これまでの検出経路（cli.step_fetch/step_backfill の check_all）は、その回に
新規取得したバッチ（日次history_days本、既定400本のうち再取得対象のみ）だけを
対象にしており、storeとして完全にマージされた系列を横断的に検査したことは
一度も無かった（T-402、9900.Tの原因調査で判明）。本スクリプトはその欠落を埋め、
storeの現在の状態全体に対して一度だけ横断検査する。

探すのは改訂履歴（revisions.csv.gz、直近の改訂のみ記録・9900.Tのケースで
observed_on=2026-08-25〜08-27の3日分しかカバーしていないことが判明）ではなく、
系列内の段差そのもの（Splitsイベントの記録なしに分割・併合比率どおりの跳びが
ある銘柄）。adjust.COMMON_RATIOS をそのまま使う（本番のsuspected_unrecorded_split
判定と同じ基準。1.5等の端数分割はadjust.py既存のコメントの理由により対象外
のまま——通常の日次変動と区別しにくく誤検出が増えるため）。

store・replayは一切変更しない読み取り専用。検出のみで自動修正はしない
（check_splitsはunadjusted_splitを検出すると修正済みdfを返すが、本スクリプトは
それをstoreへ書き戻さない）。
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> int:
    from stockbot.config import Settings
    from stockbot.data import adjust
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    print(f"[check-splits-full] 検査対象: {len(ohlcv)}銘柄"
          f"（storeの完全マージ済み系列、単発fetchの窓に限らない）")
    _fixed, issues = adjust.check_all(ohlcv)

    if len(issues) == 0:
        print("[check-splits-full] issue なし")
        return 0

    by_kind = issues["kind"].value_counts()
    print(f"[check-splits-full] issue内訳: {by_kind.to_dict()}")

    suspected = issues[issues["kind"] == "suspected_unrecorded_split"]
    print(f"\n=== suspected_unrecorded_split"
          f"（Splitsイベント記録なしの段差。{suspected['ticker'].nunique()}銘柄・{len(suspected)}件）===")
    for ticker, grp in suspected.groupby("ticker"):
        ratios = sorted(grp["ratio"].unique())
        dates = pd.to_datetime(grp["date"])
        print(f"  {ticker}: 比率{ratios} 該当日数={len(grp)} "
              f"({dates.min().date()} 〜 {dates.max().date()})")

    unadjusted = issues[issues["kind"] == "unadjusted_split"]
    if len(unadjusted):
        print(f"\n=== unadjusted_split（記録済みだがstore内で未調整。"
              f"{unadjusted['ticker'].nunique()}銘柄）===")
        for ticker, grp in unadjusted.groupby("ticker"):
            print(f"  {ticker}: {len(grp)}件")

    small = issues[issues["kind"] == "small_ratio_split"]
    if len(small):
        print(f"\n=== small_ratio_split（比率1.0近傍、検査対象外として記録のみ。"
              f"{small['ticker'].nunique()}銘柄）===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
