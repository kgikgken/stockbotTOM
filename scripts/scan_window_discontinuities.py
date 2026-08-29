"""既生成replay（頑健性窓チャンク1〜7）と重なる、全履歴再取得を行った銘柄について、
2016-02-03〜2021-07-31の範囲で系列内の段差（adjust.check_splitsと同じロジック）を
スキャンし、3分類する（診断専用。TASKS.mdのタスクではない）。

全履歴再取得（history_full_days=2600本）はマージから置換に変更したが、既に
生成済みのチャンク1〜7replayは「置換前」のstore値を使って計算されている。
再取得で書き換わった銘柄がこのreplayに登場する場合、生成時のstore値と現在の
store値が食い違っている可能性がある（比率系特徴量は理論上相殺するはずだが、
このセッションで2回連続で外れた予測と同じ形の推論のため、一様性の前提を
直接測る）。

分類:
  1. 段差が頑健性窓（2017-02-13以降）にある銘柄 → 両窓+ホールドアウトから除外
  2. 段差が2016-02-03〜02-08の先頭数行だけにある（=段差の直前がscan窓の
     ごく初期）銘柄 → 該当する既生成replay行（チャンク1〜3、日付が
     2016-02-08をlookbackに含む範囲）だけ全列再計算が必要
  3. 段差なし → 一様。対照として数銘柄を全列再計算で確認する

さらに、2016-02-08がG3の252本ルックバック窓から外れる最初のTを、9900.Tの
実際の取引日ぶんの営業日データを使って実測し、頑健性窓の開始日(2017-02-13)と
比較する（目算ではなく実測）。

store・replayは一切変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SCAN_START = pd.Timestamp("2016-02-03")
SCAN_END = pd.Timestamp("2021-07-31")
ROBUSTNESS_START = pd.Timestamp("2017-02-13")  # G3計算可能日（頑健性窓の実際の開始日）
LEADING_EDGE_CUTOFF = pd.Timestamp("2016-02-10")  # この日より前を「先頭数行」とみなす


def _g3_window_exit_date(close: pd.Series, probe_date: pd.Timestamp, lookback: int = 252) -> "pd.Timestamp | None":
    """probe_date が [T-(lookback-1), T] の窓から外れる最初の T を返す
    （実際の取引日の並びを使う。目算ではなく実測）。
    """
    idx = close.index
    if probe_date not in idx:
        return None
    pos0 = idx.get_loc(probe_date)
    exit_pos = pos0 + lookback
    if exit_pos >= len(idx):
        return None
    return idx[exit_pos]


def main() -> int:
    import argparse

    from stockbot.data import adjust
    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long

    ap = argparse.ArgumentParser(prog="python scripts/scan_window_discontinuities.py")
    ap.add_argument("--tickers", required=True, help="カンマ区切りのticker一覧")
    args = ap.parse_args()

    tickers = sorted(set(t.strip() for t in args.tickers.split(",") if t.strip()))
    print(f"[scan-window] 対象銘柄数: {len(tickers)}")
    print(f"[scan-window] スキャン範囲: {SCAN_START.date()} 〜 {SCAN_END.date()}")

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    # G3(252本)窓からの離脱日を9900.Tの実データで実測
    if "9900.T" in ohlcv and len(ohlcv["9900.T"]):
        exit_date = _g3_window_exit_date(ohlcv["9900.T"]["Close"], pd.Timestamp("2016-02-08"), 252)
        print(f"\n[g3-window] 9900.Tの実データ: 2016-02-08 が G3(252本)窓から外れる最初の T = "
              f"{exit_date.date() if exit_date is not None else '算出不能'}")
        print(f"[g3-window] 頑健性窓の開始日(2017-02-13)と比較: "
              f"{'開始日より前(=チャンク1以降には影響しない)' if exit_date is not None and exit_date < ROBUSTNESS_START else '開始日以降(=影響する可能性がある)' if exit_date is not None else '不明'}")
        # SMA200(200本)は252本よりさらに短い窓のため参考値として併記
        exit_date_sma200 = _g3_window_exit_date(ohlcv["9900.T"]["Close"], pd.Timestamp("2016-02-08"), 200)
        print(f"[g3-window] 参考(SMA200=200本): {exit_date_sma200.date() if exit_date_sma200 is not None else '算出不能'}")
    else:
        print("\n[g3-window] 9900.Tのデータが無い")

    bucket1, bucket2, bucket3 = [], [], []
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) == 0:
            bucket3.append(ticker)
            continue
        window = df[(df.index >= SCAN_START) & (df.index <= SCAN_END)]
        if len(window) < 8:  # check_splitsの最小行数要件(2*window+2, window=3)を満たさない
            bucket3.append(ticker)
            continue
        _fixed, issues = adjust.check_splits(window, ticker=ticker)
        suspected = [i for i in issues if i["kind"] == "suspected_unrecorded_split"]
        if not suspected:
            bucket3.append(ticker)
            continue
        dates = sorted(i["date"] for i in suspected)
        first_date = pd.Timestamp(dates[0])
        if first_date >= ROBUSTNESS_START:
            bucket1.append((ticker, dates))
        elif first_date < LEADING_EDGE_CUTOFF:
            bucket2.append((ticker, dates))
        else:
            # どちらのバケットにも当てはまらない(想定外)。個別確認が必要として報告
            bucket1.append((ticker, dates))

    print(f"\n=== 分類1: 段差が頑健性窓(2017-02-13以降)にある（{len(bucket1)}銘柄） ===")
    for ticker, dates in bucket1:
        print(f"  {ticker}: {[d.date().isoformat() for d in dates]}")

    print(f"\n=== 分類2: 段差が2016-02-03〜02-08の先頭数行だけにある（{len(bucket2)}銘柄） ===")
    for ticker, dates in bucket2:
        print(f"  {ticker}: {[d.date().isoformat() for d in dates]}")

    print(f"\n=== 分類3: 段差なし（{len(bucket3)}銘柄） ===")
    print(f"  {bucket3[:20]}{'...' if len(bucket3) > 20 else ''}")

    print(f"\n[scan-window] 集計: 分類1={len(bucket1)} / 分類2={len(bucket2)} / 分類3={len(bucket3)} / 合計={len(tickers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
