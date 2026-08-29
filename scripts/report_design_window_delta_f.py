"""設計窓（2021-08-01〜2024-01-31）のΔF・押し目プール全体のmean(r_h)を報告する
（2026-08-29 指示）。ΔF = mean(F通過) − mean(全押し目) なので、各F条件の寄与を
正しく読むにはプール平均も同時に見る必要がある（末端分位が負かどうかではなく、
プール平均を上回るか下回るかがΔFに効く）。

h=3（凍結後のh。ただし本スクリプトの数値は§10.3の第1関門そのものではない——
第1関門は内部確認窓で判定する。ここは設計窓での確認・記録用）と、h=10
（記録のみ、判定には使わない）の両方を出す。採否・解釈はしない（診断専用。
TASKS.md のタスクではない）。

store・data/replay のいずれも変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import pandas as pd

DESIGN_WINDOW = (pd.Timestamp("2021-08-01"), pd.Timestamp("2024-01-31"))


def main(argv=None) -> int:
    import argparse

    from stockbot.validation import layer1
    from stockbot.validation.replay import load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/report_design_window_delta_f.py")
    ap.add_argument("--replay-dir", required=True, help="data/replay（主評価窓）")
    args = ap.parse_args(argv)

    pool = load_replay_table(args.replay_dir)
    dates = pd.to_datetime(pool["date"])
    pool = pool[(dates >= DESIGN_WINDOW[0]) & (dates <= DESIGN_WINDOW[1])].reset_index(drop=True)
    pool = layer1.exclude_data_quality_tickers(pool)

    print(f"[design-delta-f] 設計窓 {DESIGN_WINDOW[0].date()}〜{DESIGN_WINDOW[1].date()}: "
          f"{len(pool)}行・{pool['date'].nunique() if len(pool) else 0}日")

    for h, note in ((3, "凍結後のh。第1関門そのものではない（内部確認窓で別途判定）"),
                   (10, "記録のみ。判定には使わない")):
        baseline_c = layer1.baseline_c_equal_weight(pool, h=h)
        summary = layer1.delta_f_summary(pool, h=h, pool_days=layer1.F_POOL_DAYS)
        print(f"\n--- h={h}（{note}） ---")
        print(f"押し目プール全体 mean(r_{h}) = {baseline_c['mean_r']:.6f}")
        print(f"ΔF_{h}: mean={summary['delta_f_mean']:.6f}  se={summary['delta_f_se']:.6f}  "
              f"t={summary['delta_f_t']:.3f}  p={summary['delta_f_p']:.4f}  "
              f"n_days_used={summary['n_days_used']}/{summary['n_days_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
