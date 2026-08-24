"""replay 実行結果の要約レポート（診断用スクリプト。TASKS.md のタスクではない）。

data/replay 配下の replay_YYYY-MM-DD.csv.gz を集計し、次を報告する:
  - 保存されている日次ファイル数・うち空（採点対象0件）のファイル数
  - 延べ行数（銘柄×日）
  - 1日あたりの採点銘柄数の平均（空でない日のみで平均）
  - D2（d2_rs60/d2_rs120）が欠損（NaN）になっている行の割合
    （pipeline.compute_daily_features が idx_close を close.index に reindex する
    ようになった際の副作用: 指数側に対応する日付が無いと NaN になる。銘柄と指数の
    営業日集合はほぼ一致するはずなので、この割合が数%を超える場合は整列に別の
    問題がある可能性がある）

スコアリング・本番パイプラインには一切影響しない読み取り専用の診断ツール。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from stockbot.validation.replay import load_replay_table


def build_summary(table: pd.DataFrame) -> Dict[str, object]:
    total_rows = int(len(table))
    if total_rows == 0:
        return {
            "total_rows": 0, "n_days_with_rows": 0, "avg_scored_per_day": None,
            "d2_rs60_missing_rate": None, "d2_rs120_missing_rate": None,
        }
    by_date = table.groupby("date").size()
    n_days_with_rows = int(len(by_date))
    avg_scored_per_day = float(by_date.mean())

    summary: Dict[str, object] = {
        "total_rows": total_rows,
        "n_days_with_rows": n_days_with_rows,
        "avg_scored_per_day": avg_scored_per_day,
    }
    for col in ("d2_rs60", "d2_rs120"):
        if col in table.columns:
            summary[f"{col}_missing_rate"] = float(table[col].isna().mean())
        else:
            summary[f"{col}_missing_rate"] = None
    return summary


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python scripts/replay_run_summary.py")
    ap.add_argument("--replay-dir", required=True)
    args = ap.parse_args(argv)

    table = load_replay_table(args.replay_dir)
    summary = build_summary(table)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    n_files = len(list(Path(args.replay_dir).glob("replay_*.csv.gz")))
    n_empty_files = sum(1 for p in Path(args.replay_dir).glob("replay_*.csv.gz")
                        if len(pd.read_csv(p)) == 0)
    print(f"[replay-summary] 保存ファイル数={n_files} / 空ファイル数={n_empty_files}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
