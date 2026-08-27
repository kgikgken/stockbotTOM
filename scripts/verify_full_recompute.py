"""既存replayアーティファクトの構造系特徴量（Lp/H0/depth/ATR/state/ゲート）＋
D1〜D8全特徴量が、PR #36後のコードで再計算しても一致するかを確認する
（診断専用。TASKS.md のタスクではない。2026-08-27 実行条件の指示への対応:
「no-op検証をD2の3列から全列に拡大」）。

verify_d2_recompute.py は d2_rs60/rs120/rsline_pos の3列だけを直接の式で
再計算する軽量な検証だった。ここでは T-205 のハーネス（tests/test_lookahead.py
run_pipeline）と同じ経路（indicators→swings→pullback→dimensions→gates）で
1銘柄ぶんの計算をフルに再現し、DAILY_FEATURES_COLS のうち価格系列から
独立に計算できる列（STRUCT_COLS・GATE_COLS・dimensions.FEATURE_IDS。
プール正規化が必要な dim_*_score / score_v1〜v3 は対象外）を比較する。

コスト: 1行あたり約35ms（swings検出のO(n)コストが支配的）。362,461行を
全件行うと約3.5時間かかるため、既知のd2不一致銘柄（9900.T）は全行、
それ以外の銘柄は1行だけサンプリングする層化サンプリングで行う
（全銘柄を横断的にカバーしつつ、既知の不一致銘柄は徹底的に検査する）。

store・replayのいずれも変更しない（読み取りのみ）。
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

STRUCT_COMPARE_COLS = [
    "state", "h0_high", "l0_low", "lp_value", "r", "leg", "leg_bars", "d",
    "depth_pct", "depth_atr", "retrace", "position", "dev5", "is_shallow", "is_deep",
]
GATE_COMPARE_COLS = ["g1", "g2", "g3", "gate_pass"]
# D8のうち regime/breadth_75/breadth_200（全銘柄横断のbreadth計算が必要）と
# earnings_days（決算予定日データが必要）は、このスクリプトが銘柄単体でしか
# 再計算しないため対象外にする（本検証の関心である「価格系列から独立に計算
# できる構造系特徴量」には該当しない）
EXCLUDED_FEATURE_IDS = {"regime", "breadth_75", "breadth_200", "earnings_days"}


def _values_equal(a, b, atol: float = 1e-9) -> bool:
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na or b_na:
        return bool(a_na) == bool(b_na)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return abs(float(a) - float(b)) <= atol
    except (TypeError, ValueError):
        return a == b


def _recompute_row(ohlcv_full: Dict[str, pd.DataFrame], idx_close_full: pd.Series,
                   ticker: str, date_t: pd.Timestamp, k: int = 3, label_n: int = 15):
    from stockbot.features import dimensions, gates, indicators, pullback, swings

    df = ohlcv_full.get(ticker)
    if df is None or date_t not in df.index:
        return None
    df = df[df.index <= date_t]
    t_pos = len(df) - 1
    open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
    volume, dividends = df["Volume"], df["Dividends"]

    sma5 = indicators.sma(close, 5)
    sma75 = indicators.sma(close, 75)
    sma200 = indicators.sma(close, 200)
    atr14 = indicators.atr_wilder(high, low, close, 14)
    raw = swings.detect_raw_swings(high, low, k)
    alt = swings.alternate_swings(raw)
    pb = pullback.pullback_state(high, low, close, sma5, sma200, atr14, alt, t_pos, k)

    idx_trunc = idx_close_full[idx_close_full.index <= date_t]
    idx_aligned = idx_trunc.reindex(close.index).ffill(limit=3)
    feats, _extra = dimensions.compute_dimensions(
        open_, high, low, close, volume, dividends, alt, pb, t_pos, k, idx_close=idx_aligned)

    gate = gates.evaluate_gates(close, high, sma75, sma200, t_pos, True, label_n)

    out = dict(pb)
    for _, row in feats.iterrows():
        out[row["id"]] = row["value"]
    out.update(gate)
    return out


def _compare(table_row: pd.Series, recomputed: dict, cols: List[str]) -> List[str]:
    mismatched = []
    for col in cols:
        if col not in recomputed:
            continue
        if not _values_equal(table_row.get(col), recomputed[col]):
            mismatched.append(col)
    return mismatched


def main() -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long
    from stockbot.validation.replay import HOLDOUT_WINDOW, load_replay_table
    from stockbot.features import dimensions

    ap = argparse.ArgumentParser(prog="python scripts/verify_full_recompute.py")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--focus-ticker", default="9900.T",
                    help="この銘柄は全行、他は1行サンプリングする")
    args = ap.parse_args()

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    idx_close_full = ohlcv.pop(IDX_TICKER)["Close"]
    idx_close_full = idx_close_full[idx_close_full.index < HOLDOUT_WINDOW[0]]
    for t in ohlcv:
        ohlcv[t] = ohlcv[t][ohlcv[t].index < HOLDOUT_WINDOW[0]]

    table = load_replay_table(args.replay_dir)
    table["date"] = pd.to_datetime(table["date"])
    compare_cols = (STRUCT_COMPARE_COLS + GATE_COMPARE_COLS
                    + [f for f in dimensions.FEATURE_IDS if f not in EXCLUDED_FEATURE_IDS])

    focus_rows = table[table["ticker"] == args.focus_ticker]
    other_tickers = table[table["ticker"] != args.focus_ticker]
    sample_idx = other_tickers.groupby("ticker").apply(
        lambda g: g.index[len(g) // 2]).to_numpy()
    sampled_others = other_tickers.loc[sample_idx]

    print(f"検証対象: {args.focus_ticker} 全{len(focus_rows)}行 + "
          f"他{other_tickers['ticker'].nunique()}銘柄から各1行サンプリング")

    t_start = time.monotonic()
    n_checked = 0
    mismatch_summary: Dict[str, int] = {}
    mismatch_examples = []

    def _check(rows: pd.DataFrame, label: str):
        nonlocal n_checked
        for _, row in rows.iterrows():
            ticker, date_t = row["ticker"], row["date"]
            recomputed = _recompute_row(ohlcv, idx_close_full, ticker, date_t)
            n_checked += 1
            if recomputed is None:
                continue
            mismatched = _compare(row, recomputed, compare_cols)
            for col in mismatched:
                mismatch_summary[col] = mismatch_summary.get(col, 0) + 1
            if mismatched and len(mismatch_examples) < 30:
                mismatch_examples.append((label, ticker, date_t.date(), mismatched))

    _check(focus_rows, args.focus_ticker)
    _check(sampled_others, "sampled_other")

    elapsed = time.monotonic() - t_start
    print(f"検証済み行数: {n_checked}（{elapsed:.0f}秒）")
    if mismatch_summary:
        print("列別の不一致件数:")
        for col, n in sorted(mismatch_summary.items(), key=lambda x: -x[1]):
            print(f"  {col}: {n} 件")
        print("不一致の例:")
        for label, ticker, date_, cols in mismatch_examples:
            print(f"  [{label}] {ticker} {date_}: {cols}")
    else:
        print("[OK] 不一致なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
