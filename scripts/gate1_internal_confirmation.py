"""DESIGN.md §10.3 第1関門（内部確認窓 2024-02-01〜2026-01-30）を判定する。

条件（事前登録済み・R1）: W1 の ΔF が正、かつ (c) が (a′) を上回る。h=3（§9.3 で
設計窓のみから選択・凍結済み）。

2026-08-30 指摘への対応: data/a_prime は日付ベースのファイル名でキャッシュされており、
設計窓（2021-08-01〜2024-01-31）と内部確認窓（2024-02-01〜2026-01-30）の両方の出力が
同じキャッシュアーティファクトに蓄積されうる。したがって data/a_prime を無条件で
全件読み込むと設計窓の613日分が内部確認窓の集計に混入する。select_h.py・
report_design_window_delta_f.py と同じパターンで、(c)・(a′) の両方を
INTERNAL_CONFIRMATION_WINDOW で明示的に絞り込んでから集計する（下の窓フィルタが
唯一の新規ロジック。それ以外は既存のテスト済み layer1 関数を呼ぶだけ）。

判定条件そのものは §10.3 で事前登録済みの機械的規則であり、真偽を計算して表示する
だけで、通過/不通過が意味することの解釈（採否・継続・撤退の判断）はしない
（CLAUDE.md「検証結果を解釈しない」。判断は設計責任者）。

store・data/replay・data/a_prime のいずれも変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import pandas as pd

INTERNAL_CONFIRMATION_WINDOW = (pd.Timestamp("2024-02-01"), pd.Timestamp("2026-01-30"))


def _filter_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if len(df) == 0 or "date" not in df.columns:
        return df
    dates = pd.to_datetime(df["date"])
    return df[(dates >= start) & (dates <= end)].reset_index(drop=True)


def main(argv=None) -> int:
    import argparse

    from stockbot.validation import layer1
    from stockbot.validation.replay import load_a_prime_table, load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/gate1_internal_confirmation.py")
    ap.add_argument("--replay-dir", required=True, help="data/replay（主評価窓、(c)の母集団）")
    ap.add_argument("--a-prime-dir", required=True, help="data/a_prime（(a′)の出力先）")
    args = ap.parse_args(argv)

    start, end = INTERNAL_CONFIRMATION_WINDOW
    pool = load_replay_table(args.replay_dir)
    a_prime = load_a_prime_table(args.a_prime_dir)

    pool = _filter_window(pool, start, end)
    a_prime = _filter_window(a_prime, start, end)

    pool = layer1.exclude_data_quality_tickers(pool)
    a_prime = layer1.exclude_data_quality_tickers(a_prime)

    print(f"[gate1] 内部確認窓 {start.date()}〜{end.date()}: "
          f"(c) pool {len(pool)}行・{pool['date'].nunique() if len(pool) else 0}日 / "
          f"(a′) {len(a_prime)}行・{a_prime['date'].nunique() if len(a_prime) else 0}日")

    if len(pool) == 0 or len(a_prime) == 0:
        print("[gate1] (c) または (a′) が空のため判定不能")
        return 1

    h = layer1.DEFAULT_H
    delta_f = layer1.delta_f_summary(pool, h=h, pool_days=layer1.F_POOL_DAYS)
    c_vs_a = layer1.c_minus_a_prime_summary(pool, a_prime, h=h)

    print(f"\n--- h={h} ---")
    print(f"W1 ΔF_{h}: mean={delta_f['delta_f_mean']:.6f}  se={delta_f['delta_f_se']:.6f}  "
          f"t={delta_f['delta_f_t']:.3f}  p={delta_f['delta_f_p']:.4f}  "
          f"n_days_used={delta_f['n_days_used']}/{delta_f['n_days_total']}")
    print(f"(c)-(a′)_{h}: mean={c_vs_a['c_minus_a_prime_mean']:.6f}  "
          f"se={c_vs_a['c_minus_a_prime_se']:.6f}  t={c_vs_a['c_minus_a_prime_t']:.3f}  "
          f"p={c_vs_a['c_minus_a_prime_p']:.4f}  "
          f"n_days_used={c_vs_a['n_days_used']}/{c_vs_a['n_days_total']}")

    delta_f_positive = bool(pd.notna(delta_f["delta_f_mean"]) and delta_f["delta_f_mean"] > 0)
    c_above_a_prime = bool(pd.notna(c_vs_a["c_minus_a_prime_mean"]) and c_vs_a["c_minus_a_prime_mean"] > 0)
    gate1_pass = delta_f_positive and c_above_a_prime

    print(f"\n[gate1] 条件1（W1のΔFが正）: {delta_f_positive}")
    print(f"[gate1] 条件2（(c)が(a′)を上回る）: {c_above_a_prime}")
    print(f"[gate1] 第1関門: {'通過' if gate1_pass else '不通過'}"
          "（事前登録済みの機械的規則を適用した結果。解釈・採否判断はしない）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
