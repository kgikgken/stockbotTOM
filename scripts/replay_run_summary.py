"""replay 実行結果の要約レポート（診断用スクリプト。TASKS.md のタスクではない）。

data/replay 配下の replay_YYYY-MM-DD.csv.gz を集計し、次を報告する:
  - 保存されている日次ファイル数・うち空（採点対象0件）のファイル数
    （2026-08-26 追加: 空ファイルを「データ無し」「候補0件（ゲート等）」の内訳で分ける。
    replay_meta_YYYY-MM-DD.json サイドカーが無い空ファイルは「データ無し」扱いにする
    ——このガードの追加前に生成された空ファイルとの後方互換のための安全側デフォルト）
  - 延べ行数（銘柄×日）
  - 1日あたりの採点銘柄数の平均（空でない日のみで平均）
  - DESIGN.md §6.1 の F 除外14条件が参照する12特徴量それぞれの欠損（NaN）率
    （D2の d2_rs60/d2_rs120/d2_rsline_pos は、pipeline.compute_daily_features が
    idx_close を close.index に reindex するようになった際の副作用: 指数側に
    対応する日付が無いと NaN になる。銘柄と指数の営業日集合はほぼ一致するはずなので、
    この割合が数%を超える場合は整列に別の問題がある可能性がある）
  - table に warmup 列がある場合（頑健性窓）は、既定で warmup=True の行を集計から
    除外する（DESIGN.md §10.1 開始日の規則。プール継続のためだけに計算した行であり、
    集計に含めない）

スコアリング・本番パイプラインには一切影響しない読み取り専用の診断ツール。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from stockbot.validation.replay import (
    EMPTY_REASON_NO_CANDIDATES, EMPTY_REASON_NO_DATA, load_replay_table,
)

# DESIGN.md §6.1: F1〜F14 が参照する特徴量（重複を除くと12個）
F_ID_TO_COLUMN = {
    "F1": "d3_depth_pct", "F2": "d3_depth_pct",
    "F3": "d3_position",
    "F4": "d3_dev5",
    "F5": "d3_maxdrop",
    "F6": "d3_retrace",
    "F7": "d3_hl_dist",
    "F8": "d3_ma_dist",
    "F9": "d3_depth_atr",
    "F10": "d2_rs60",
    "F11": "d2_rs120",
    "F12": "d2_rsline_pos",
    "F13": "d4_pb_ratio",
    "F14": "d5_atr_ratio",
}


def build_summary(table: pd.DataFrame, exclude_warmup: bool = True) -> Dict[str, object]:
    if exclude_warmup and "warmup" in table.columns:
        n_warmup = int(table["warmup"].fillna(False).astype(bool).sum())
        table = table[~table["warmup"].fillna(False).astype(bool)]
    else:
        n_warmup = 0

    total_rows = int(len(table))
    if total_rows == 0:
        return {
            "total_rows": 0, "n_days_with_rows": 0, "avg_scored_per_day": None,
            "n_warmup_rows_excluded": n_warmup,
            "f_missing_rates": {fid: None for fid in F_ID_TO_COLUMN},
        }
    by_date = table.groupby("date").size()
    n_days_with_rows = int(len(by_date))
    avg_scored_per_day = float(by_date.mean())

    summary: Dict[str, object] = {
        "total_rows": total_rows,
        "n_days_with_rows": n_days_with_rows,
        "avg_scored_per_day": avg_scored_per_day,
        "n_warmup_rows_excluded": n_warmup,
    }
    f_missing = {}
    for fid, col in F_ID_TO_COLUMN.items():
        f_missing[fid] = {
            "column": col,
            "missing_rate": float(table[col].isna().mean()) if col in table.columns else None,
        }
    summary["f_missing_rates"] = f_missing
    return summary


def by_year_summary(table: pd.DataFrame, exclude_warmup: bool = True) -> pd.DataFrame:
    """年別の1日あたり行数（診断・記録用。2026-08-29 指示: 頑健性窓中盤の密度低下を
    年別に確認する）。"""
    if exclude_warmup and "warmup" in table.columns:
        table = table[~table["warmup"].fillna(False).astype(bool)]
    cols = ["year", "n_days", "total_rows", "avg_rows_per_day"]
    if len(table) == 0:
        return pd.DataFrame(columns=cols)
    years = pd.to_datetime(table["date"]).dt.year
    rows = []
    for year, idx in table.groupby(years).groups.items():
        sub = table.loc[idx]
        by_date = sub.groupby("date").size()
        rows.append({"year": int(year), "n_days": int(len(by_date)), "total_rows": int(len(sub)),
                    "avg_rows_per_day": float(by_date.mean())})
    return pd.DataFrame(rows, columns=cols).sort_values("year").reset_index(drop=True)


def empty_file_breakdown(replay_dir: Path) -> Dict[str, int]:
    """空ファイルを「データ無し」「候補0件（ゲート等）」で分けて数える
    （replay_meta_*.json サイドカーを参照。無ければ安全側でデータ無し扱い）。"""
    replay_dir = Path(replay_dir)
    n_no_data = n_no_candidates = 0
    for p in sorted(replay_dir.glob("replay_*.csv.gz")):
        if len(pd.read_csv(p)) != 0:
            continue
        date_str = p.name[len("replay_"):-len(".csv.gz")]
        meta_path = replay_dir / f"replay_meta_{date_str}.json"
        reason = EMPTY_REASON_NO_DATA
        if meta_path.exists():
            try:
                reason = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "empty_reason", EMPTY_REASON_NO_DATA)
            except (ValueError, OSError):
                pass
        if reason == EMPTY_REASON_NO_CANDIDATES:
            n_no_candidates += 1
        else:
            n_no_data += 1
    return {"no_data": n_no_data, "no_candidates": n_no_candidates}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python scripts/replay_run_summary.py")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--include-warmup", action="store_true",
                    help="warmup=True の行も集計に含める（既定は除外。診断用）")
    args = ap.parse_args(argv)

    table = load_replay_table(args.replay_dir)
    summary = build_summary(table, exclude_warmup=not args.include_warmup)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    by_year = by_year_summary(table, exclude_warmup=not args.include_warmup)
    print("[replay-summary] 年別:")
    print(by_year.to_string(index=False))

    n_files = len(list(Path(args.replay_dir).glob("replay_*.csv.gz")))
    breakdown = empty_file_breakdown(Path(args.replay_dir))
    n_empty_files = breakdown["no_data"] + breakdown["no_candidates"]
    print(f"[replay-summary] 保存ファイル数={n_files} / 空ファイル数={n_empty_files} "
          f"(データ無し={breakdown['no_data']} / 候補0件・ゲート等={breakdown['no_candidates']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
