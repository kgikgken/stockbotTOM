"""全履歴再取得（refetch-recent-splits）を行った銘柄が、既に生成済みのreplay
（頑健性窓チャンク1〜7など）に何行含まれているかを数える（診断専用。TASKS.mdの
タスクではない）。

全履歴再取得はstore側の該当銘柄の過去データを丸ごと今日時点の値へ差し替える。
既に生成済みのreplayは生成当時のstore値を使って計算済みのため、再取得後の
storeと食い違う可能性がある（比率系の特徴量は理論上相殺するはずだが、この
セッションで2回連続で外れた予測と同じ形の推論のため、実測で確認する）。

0件ならreplayとstoreの食い違いは実害が無い（対象銘柄がreplayに登場しない）。
非0件ならその銘柄について全列再計算（verify_full_recompute.py）で一致を確認
する必要がある。store・replayのいずれも変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> int:
    import argparse

    from stockbot.validation.replay import load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/count_refetched_ticker_rows_in_replay.py")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--tickers", required=True,
                    help="カンマ区切りのticker一覧（全履歴再取得を行った銘柄）")
    args = ap.parse_args()

    tickers = sorted(set(t.strip() for t in args.tickers.split(",") if t.strip()))
    print(f"[count-refetched] 対象銘柄数: {len(tickers)}")

    table = load_replay_table(args.replay_dir)
    if len(table) == 0:
        print("[count-refetched] replayが空")
        return 0

    hit = table[table["ticker"].isin(tickers)]
    print(f"[count-refetched] replay総行数: {len(table)} / 再取得銘柄と重なる行数: {len(hit)}")

    if len(hit) == 0:
        print("[count-refetched] 重なりゼロ。再取得によるreplayへの影響なし")
        return 0

    counts = hit.groupby("ticker").agg(
        n_rows=("date", "size"),
        min_date=("date", "min"),
        max_date=("date", "max"),
    ).sort_values("n_rows", ascending=False)
    print(f"[count-refetched] 重なりのある銘柄数: {len(counts)}")
    for ticker, row in counts.iterrows():
        print(f"  {ticker}: {row['n_rows']}行 "
              f"({pd.Timestamp(row['min_date']).date()} 〜 {pd.Timestamp(row['max_date']).date()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
