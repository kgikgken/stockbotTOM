"""分割予定が公表済み・効力発生前の銘柄が、store に「先出し」で分割調整された
価格を持っていないかを全銘柄について点検する（診断専用。TASKS.md のタスクでは
ない。9900.T の d2 no-op不一致調査への対応、2026-08-27）。

9900.T（サガミホールディングス）のケースで判明した実際の失敗モード: yfinance が
株式分割の効力発生前に、取得済みの全期間（Stock Splits 列にイベント記録が付く
はずの日を含む）に対して分割係数を先出しで一律適用することがある。この場合、
系列内の連続バー間には段差が生じない（全期間が一律にスケールされるため）ので
adjust.py の check_splits（連続バー間の跳びを見る）はこれを検出できない
（SPEC §4「イベント記録なしに分割比率どおりの跳びがある銘柄は未記録分割の疑い
として除外」は、系列内に段差がある場合のみ有効）。

唯一の実際の痕跡は data/store/revisions.csv.gz（同じ日付の値が前回取得時と
今回取得時で変わった記録）に残る。close/old_close（またはその逆数）が
分割・併合として典型的な「きれいな比率」に近い、直近観測分をすべて拾い出す。

store を変更しない（読み取りのみ）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# adjust.py の COMMON_RATIOS と同じ（日本株で一般的な分割・併合比率）
COMMON_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 1 / 3, 0.25, 0.2, 0.1)
CLEAN_RATIO_TOL = 0.02


def _closest_clean_ratio(ratio: float):
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    for r in COMMON_RATIOS:
        if abs(ratio - r) / r <= CLEAN_RATIO_TOL:
            return r
    return None


def main() -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import OhlcvStore

    ap = argparse.ArgumentParser(prog="python scripts/scan_pending_split_adjustments.py")
    ap.add_argument("--recent-days", type=int, default=3,
                    help="observed_on がこの日数以内の改訂だけを対象にする（既定3日）")
    args = ap.parse_args()

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    if not store.revisions_path.exists():
        print("[scan_pending_split_adjustments] revisions.csv.gz が無い（改訂履歴なし）")
        return 0

    rev = pd.read_csv(store.revisions_path)
    if len(rev) == 0:
        print("[scan_pending_split_adjustments] revisions.csv.gz は空")
        return 0

    rev["observed_on"] = pd.to_datetime(rev["observed_on"], format="mixed")
    rev["date"] = pd.to_datetime(rev["date"], format="mixed")
    cutoff = rev["observed_on"].max() - pd.Timedelta(days=args.recent_days)
    recent = rev[rev["observed_on"] >= cutoff].copy()
    print(f"検査対象: observed_on >= {cutoff.date()} の改訂 {len(recent)} 件"
          f"（revisions.csv.gz 全体 {len(rev)} 件中）")

    recent = recent[(recent["close"] > 0) & (recent["old_close"] > 0)]
    recent["ratio"] = recent["old_close"] / recent["close"]
    recent["clean_ratio"] = recent["ratio"].apply(_closest_clean_ratio)
    suspects = recent[recent["clean_ratio"].notna()]

    if len(suspects) == 0:
        print("[scan_pending_split_adjustments] きれいな比率の改訂は見つからなかった"
              "（9900.T型の先出し適用の疑いは無し）")
        return 0

    print(f"\n=== 分割・併合比率どおりの改訂（先出し適用の疑い）: {suspects['ticker'].nunique()} 銘柄 ===")
    for ticker, grp in suspects.groupby("ticker"):
        ratios = sorted(grp["clean_ratio"].unique())
        dates = grp["date"]
        print(f"  {ticker}: 比率{ratios} 該当日数={len(grp)} "
              f"({dates.min().date()} 〜 {dates.max().date()}) "
              f"最新observed_on={grp['observed_on'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
