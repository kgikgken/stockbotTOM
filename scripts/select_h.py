"""DESIGN.md §9.3: 保有期間 h を設計窓（2021-08-01〜2024-01-31）だけで選ぶ。
選択基準は「(c) − (a′) が最大になる h」（h ∈ {3,5,10,15,20,30}）。

(c) は data/replay（主評価窓の一部として既に生成済み）から設計窓ぶんだけを
切り出した pool、(a′) は scripts/run_a_prime_replay.py の出力（同じ設計窓で
実行しておく必要がある）。両方に10銘柄のデータ品質機械的除外（T-402）を適用する。

このスクリプトは選択基準（最大値）を機械的に適用して表と選ばれた h を出すだけで、
その値の妥当性を解釈・判断しない（CLAUDE.md「検証結果を解釈しない」）。
"""
from __future__ import annotations

import sys

import pandas as pd

DESIGN_WINDOW = (pd.Timestamp("2021-08-01"), pd.Timestamp("2024-01-31"))


def main(argv=None) -> int:
    import argparse

    from stockbot.validation import layer1
    from stockbot.validation.replay import load_a_prime_table, load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/select_h.py")
    ap.add_argument("--replay-dir", required=True, help="data/replay（主評価窓）")
    ap.add_argument("--a-prime-dir", required=True, help="scripts/run_a_prime_replay.py の出力先")
    args = ap.parse_args(argv)

    pool = load_replay_table(args.replay_dir)
    a_prime = load_a_prime_table(args.a_prime_dir)

    pool_dates = pd.to_datetime(pool["date"])
    pool = pool[(pool_dates >= DESIGN_WINDOW[0]) & (pool_dates <= DESIGN_WINDOW[1])].reset_index(drop=True)
    a_prime_dates = pd.to_datetime(a_prime["date"]) if len(a_prime) else pool_dates.iloc[0:0]
    a_prime = a_prime[(a_prime_dates >= DESIGN_WINDOW[0])
                      & (a_prime_dates <= DESIGN_WINDOW[1])].reset_index(drop=True)

    pool = layer1.exclude_data_quality_tickers(pool)
    a_prime = layer1.exclude_data_quality_tickers(a_prime)

    print(f"[select-h] 設計窓 {DESIGN_WINDOW[0].date()}〜{DESIGN_WINDOW[1].date()}: "
          f"(c) pool {len(pool)}行・{pool['date'].nunique() if len(pool) else 0}日 / "
          f"(a′) {len(a_prime)}行・{a_prime['date'].nunique() if len(a_prime) else 0}日")

    if len(pool) == 0 or len(a_prime) == 0:
        print("[select-h] (c) または (a′) が空のため中止")
        return 1

    table = layer1.select_h(pool, a_prime)
    print(table.to_string(index=False))

    selected = table[table["selected"]]
    if len(selected):
        h = int(selected.iloc[0]["h"])
        print(f"\n[select-h] 選択された h = {h}"
              f"（(c)−(a′) = {selected.iloc[0]['c_minus_a_prime_mean']:.5f}, "
              f"t = {selected.iloc[0]['c_minus_a_prime_t']:.3f}）")
    else:
        print("\n[select-h] 選択不能（全 h で (c)−(a′) が計算不能）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
