"""既存replayアーティファクトの d2_rs60/d2_rs120/d2_rsline_pos が、PR #36
（idx_close.reindex(close.index).ffill(limit=3) 追加）後のコードでも完全一致することを
確認する（診断専用。TASKS.md のタスクではない）。

CLAUDE.md §11 の「再計算一致」の考え方を、既存の保存済みアーティファクトに対して
事後的に適用する: 「T で切って計算した特徴量 = 全期間で計算して T 行を取り出した特徴量」
の一致を、実際に store のデータから再計算して検証する。

主評価窓（2021-08-01〜2026-01-30、362,461行）が新コードでも no-op（値が変わらない）
という主張は、指数データの欠落が窓内で0件であることからの推論に過ぎない。
idx_close.reindex(close.index) は「銘柄側の日付」に合わせるため、ある銘柄が指数に
無い日付を持っていれば（d2_rs120 の場合、窓の開始直後は窓の外＝2021年頭ごろまで
遡る）、指数の欠落が窓内で0件でも ffill が発火しうる。推論ではなく実測で確認する。

d2_rs60/d2_rs120/d2_rsline_pos は dimensions.py の以下の式をそのまま再現する
（history_pool やゲート等、プール正規化スコアの計算は一切不要。D2の生値は独立に
計算できるため、再生成よりはるかに軽い）:

    for n, fid in ((60, "d2_rs60"), (120, "d2_rs120")):
        c0, c1, i0, i1 = c[t_pos-n], c[t_pos], ic[t_pos-n], ic[t_pos]
        values[fid] = log(c1/c0) - log(i1/i0)
    rs_line = c[t_pos-59:t_pos+1] / ic[t_pos-59:t_pos+1]
    values["d2_rsline_pos"] = (rs_line[-1]-rmin)/(rmax-rmin)

store・backfill・replay の既存アーティファクトのいずれも変更しない（読み取りのみ）。
"""
from __future__ import annotations

import sys
from typing import Dict

import numpy as np
import pandas as pd

D2_COLS = ["d2_rs60", "d2_rs120", "d2_rsline_pos"]


def _recompute_d2(close: pd.Series, idx_close_full: pd.Series, date_t: pd.Timestamp) -> Dict[str, float]:
    """dimensions.py の D2 の式をそのまま再現する（現行コード: pipeline.py が
    idx_close.reindex(close.index).ffill(limit=3) を渡す前提）。"""
    close_trunc = close[close.index <= date_t]
    if len(close_trunc) == 0 or close_trunc.index[-1] != date_t:
        return {c: np.nan for c in D2_COLS}
    idx_trunc = idx_close_full[idx_close_full.index <= date_t]
    idx_aligned = idx_trunc.reindex(close_trunc.index).ffill(limit=3)

    c = close_trunc.to_numpy(dtype=float)
    ic = idx_aligned.to_numpy(dtype=float)
    t_pos = len(c) - 1
    out: Dict[str, float] = {c_: np.nan for c_ in D2_COLS}
    for n, fid in ((60, "d2_rs60"), (120, "d2_rs120")):
        if t_pos - n >= 0:
            c0, c1, i0, i1 = c[t_pos - n], c[t_pos], ic[t_pos - n], ic[t_pos]
            if min(c0, c1, i0, i1) > 0 and not np.isnan([c0, c1, i0, i1]).any():
                out[fid] = float(np.log(c1 / c0) - np.log(i1 / i0))
    window = 60
    if t_pos - window + 1 >= 0:
        rs_line = c[t_pos - window + 1: t_pos + 1] / ic[t_pos - window + 1: t_pos + 1]
        if not np.isnan(rs_line).any():
            rmin, rmax = rs_line.min(), rs_line.max()
            if rmax > rmin:
                out["d2_rsline_pos"] = float((rs_line[-1] - rmin) / (rmax - rmin))
    return out


def _values_match(a: float, b: float, atol: float = 1e-9) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= atol


def main() -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long
    from stockbot.validation.replay import HOLDOUT_WINDOW, load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/verify_d2_recompute.py")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--max-mismatches-to-show", type=int, default=20)
    args = ap.parse_args()

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_close_full = ohlcv.pop(IDX_TICKER)["Close"]
    idx_close_full = idx_close_full[idx_close_full.index < HOLDOUT_WINDOW[0]]

    table = load_replay_table(args.replay_dir)
    if len(table) == 0:
        print(f"[verify_d2_recompute] {args.replay_dir} にreplayデータが無い")
        return 1
    table["date"] = pd.to_datetime(table["date"])
    print(f"検証対象: {len(table)} 行（{table['ticker'].nunique()} 銘柄、"
          f"{table['date'].min().date()} 〜 {table['date'].max().date()}）")

    n_checked = 0
    n_mismatch = {c: 0 for c in D2_COLS}
    mismatches = []
    for ticker, rows in table.groupby("ticker"):
        close = ohlcv.get(ticker)
        if close is None or len(close) == 0:
            print(f"[WARN] {ticker}: storeに無い（比較不能。{len(rows)}行スキップ）")
            continue
        close = close["Close"]
        close = close[close.index < HOLDOUT_WINDOW[0]]
        for _, row in rows.iterrows():
            date_t = row["date"]
            recomputed = _recompute_d2(close, idx_close_full, date_t)
            n_checked += 1
            for col in D2_COLS:
                stored = row[col]
                if not _values_match(stored, recomputed[col]):
                    n_mismatch[col] += 1
                    if len(mismatches) < args.max_mismatches_to_show:
                        mismatches.append((ticker, date_t.date(), col, stored, recomputed[col]))

    print(f"検証済み行数: {n_checked}")
    for col in D2_COLS:
        print(f"  {col}: 不一致 {n_mismatch[col]} 件")
    if mismatches:
        print(f"不一致の例（最大{args.max_mismatches_to_show}件）:")
        for ticker, date_, col, stored, recomputed in mismatches:
            print(f"  {ticker} {date_} {col}: stored={stored} recomputed={recomputed}")
        return 1
    print("[OK] 全行一致。主評価窓は no-op（値が変わらない）ことを実測で確認した")
    return 0


if __name__ == "__main__":
    sys.exit(main())
