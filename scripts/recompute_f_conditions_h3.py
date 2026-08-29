"""DESIGN.md §10.2 手順1「単変量（全h）…設計窓のみで実施し、F の分位境界を確定する」
への対応（2026-08-29 指示）。これまで F1〜F14 の境界値（DESIGN.md §6.1）は h=10 の
曲線から決めており、h=3 への凍結後にこの手順を踏んでいなかった。

ここでは F の向き・閾値（DESIGN.md §5・§6.1、事前登録済み）は一切変更せず、
設計窓（2021-08-01〜2024-01-31）・h=3 で各条件が除外する末端分位の mean(r_3) の
符号だけを再集計する。向きの付け替えは§10.3の見直し回数を消費済みのため行わない
（診断専用。TASKS.md のタスクではない）。

store・data/replay のいずれも変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import pandas as pd

DESIGN_WINDOW = (pd.Timestamp("2021-08-01"), pd.Timestamp("2024-01-31"))
H = 3

F_RECHECK_COLS = ["fid", "feature_id", "tail", "threshold", "n_excluded",
                 f"mean_r_{H}_tail", "still_negative"]


def recompute_f_conditions(pool: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """F1〜F14それぞれについて、既存の閾値・方向（DESIGN.md §6.1、変更しない）で
    除外される末端分位のmean(r_h)を計算する。still_negative は依然として負なら
    True、正に転じていれば False、該当0件で判定不能なら None。
    """
    from stockbot.validation import layer1

    r_col = f"r_{h}"
    rows = []
    pctl_cache = {}
    for fid, feature_id, tail, threshold in layer1.F_CONDITIONS:
        if feature_id not in pctl_cache:
            pctl_cache[feature_id] = layer1.pool_percentile_series(pool, feature_id, pool_days=layer1.F_POOL_DAYS)
        pctl = pctl_cache[feature_id]
        raw_missing = pool[feature_id].isna() if feature_id in pool.columns \
            else pd.Series(True, index=pool.index)
        excluded = (pctl < threshold) if tail == "low" else (pctl > threshold)
        excluded = excluded.fillna(False) & ~raw_missing
        n_excluded = int(excluded.sum())
        tail_mean = float(pool.loc[excluded, r_col].mean()) if n_excluded and r_col in pool.columns \
            else float("nan")
        still_negative = bool(tail_mean < 0) if tail_mean == tail_mean else None  # NaN安全
        rows.append({"fid": fid, "feature_id": feature_id, "tail": tail, "threshold": threshold,
                    "n_excluded": n_excluded, f"mean_r_{h}_tail": tail_mean,
                    "still_negative": still_negative})
    return pd.DataFrame(rows, columns=["fid", "feature_id", "tail", "threshold", "n_excluded",
                                       f"mean_r_{h}_tail", "still_negative"])


def summarize(table: pd.DataFrame) -> dict:
    return {
        "n_still_negative": int(table["still_negative"].apply(lambda v: v is True).sum()),
        "n_flipped": int(table["still_negative"].apply(lambda v: v is False).sum()),
        "n_undetermined": int(table["still_negative"].isna().sum()),
        "n_total": int(len(table)),
    }


def main(argv=None) -> int:
    import argparse

    from stockbot.validation import layer1
    from stockbot.validation.replay import load_replay_table

    ap = argparse.ArgumentParser(prog="python scripts/recompute_f_conditions_h3.py")
    ap.add_argument("--replay-dir", required=True, help="data/replay（主評価窓）")
    args = ap.parse_args(argv)

    pool = load_replay_table(args.replay_dir)
    dates = pd.to_datetime(pool["date"])
    pool = pool[(dates >= DESIGN_WINDOW[0]) & (dates <= DESIGN_WINDOW[1])].reset_index(drop=True)
    pool = layer1.exclude_data_quality_tickers(pool)

    print(f"[f-recheck] 設計窓 {DESIGN_WINDOW[0].date()}〜{DESIGN_WINDOW[1].date()}: "
          f"{len(pool)}行・{pool['date'].nunique() if len(pool) else 0}日、h={H}")

    table = recompute_f_conditions(pool, h=H)
    print(table.to_string(index=False))

    s = summarize(table)
    print(f"\n[f-recheck] 集計: 末端分位が依然として負={s['n_still_negative']}件 / "
          f"符号反転(正に転じた)={s['n_flipped']}件 / 判定不能(該当0件)={s['n_undetermined']}件 / "
          f"合計{s['n_total']}件（h={H}）")
    print("[f-recheck] 向き（§5の用途列）は変更していない。ここで報告するのは"
          "既存の閾値・方向をh=3に適用した結果のみ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
