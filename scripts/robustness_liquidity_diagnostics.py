"""頑健性窓・主評価窓の流動性構成診断とΔF算出（診断専用。TASKS.mdのタスクではない）。

2026-08-27 チャンク4結果を受けた追加指示への対応:
  - 窓別の20日平均売買代金(ADV)分布を報告
  - ADV>=2億円部分集合でのΔFを診断として算出（判定には使わない。事前登録のみ）
  - ΔFの時系列平均から押し目状態の行数が閾値未満の日を除外し、除外日数を報告
  - 9900.Tをデータ品質により両窓の集計から除外（TASKS.md T-402）

ADVは universe.build.liquidity_stats と同じ定義（Close*Volumeの直近20営業日平均）を、
pool の各行 (ticker, date) についてその date までのデータだけを使って因果的に計算する
（未来を見ない）。store・replay を一切変更しない読み取り専用の診断ツール。
CLAUDE.md「検証結果を解釈しない」に従い、数値を出すだけで採否・良否の判断はしない。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _causal_adv20(pool: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """pool の各行 (ticker, date) について、date までの直近 window 営業日の
    平均売買代金（Close*Volume）を因果的に計算する（未来のバーは使わない）。
    """
    out = pd.Series(np.nan, index=pool.index, dtype=float)
    dates = pd.to_datetime(pool["date"])
    for ticker, idx in pool.groupby("ticker").groups.items():
        df = ohlcv.get(ticker)
        if df is None or len(df) == 0:
            continue
        turnover = (df["Close"].astype(float) * df["Volume"].astype(float)).rolling(window).mean()
        out.loc[idx] = dates.loc[idx].map(turnover)
    return out


def adv_distribution(adv: pd.Series, threshold: float) -> dict:
    valid = adv.dropna()
    if len(valid) == 0:
        return {"n": 0, "n_missing": int(adv.isna().sum())}
    return {
        "n": int(len(valid)), "n_missing": int(adv.isna().sum()),
        "min": float(valid.min()), "p10": float(valid.quantile(0.10)),
        "p25": float(valid.quantile(0.25)), "p50": float(valid.quantile(0.50)),
        "p75": float(valid.quantile(0.75)), "p90": float(valid.quantile(0.90)),
        "max": float(valid.max()),
        "n_ge_threshold": int((valid >= threshold).sum()),
        "frac_ge_threshold": float((valid >= threshold).mean()),
    }


def empty_dates(replay_dir: Path) -> list:
    """空(データ無し/候補0件)ファイルの日付一覧（replay_run_summary.pyの
    empty_file_breakdownと同じサイドカー参照ロジック）。"""
    from stockbot.validation.replay import EMPTY_REASON_NO_DATA

    out = []
    for p in sorted(Path(replay_dir).glob("replay_*.csv.gz")):
        if len(pd.read_csv(p)) != 0:
            continue
        date_str = p.name[len("replay_"):-len(".csv.gz")]
        meta_path = Path(replay_dir) / f"replay_meta_{date_str}.json"
        reason = EMPTY_REASON_NO_DATA
        if meta_path.exists():
            try:
                reason = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "empty_reason", EMPTY_REASON_NO_DATA)
            except (ValueError, OSError):
                pass
        out.append({"date": date_str, "reason": reason})
    return out


def run_window(name: str, replay_dir: Path, ohlcv: Dict[str, pd.DataFrame], h: int,
               min_rows_per_day: int, adv_threshold: float, exclude_warmup: bool) -> dict:
    from stockbot.validation import layer1
    from stockbot.validation.replay import load_replay_table

    pool = load_replay_table(replay_dir)
    if len(pool) == 0:
        return {"window": name, "n_rows": 0}

    n_warmup_excluded = 0
    if exclude_warmup and "warmup" in pool.columns:
        is_warmup = pool["warmup"].fillna(False).astype(bool)
        n_warmup_excluded = int(is_warmup.sum())
        pool = pool[~is_warmup].reset_index(drop=True)

    n_before_dq = len(pool)
    pool = layer1.exclude_data_quality_tickers(pool)
    n_excluded_dq_rows = n_before_dq - len(pool)

    n_days_total = int(pool["date"].nunique())

    delta_all = layer1.delta_f_summary(pool, h=h, min_rows_per_day=None)
    delta_filtered = layer1.delta_f_summary(pool, h=h, min_rows_per_day=min_rows_per_day)

    day_counts = pool.groupby("date").size()
    excluded_days = sorted(str(pd.Timestamp(d).date())
                           for d in day_counts[day_counts < min_rows_per_day].index)

    adv = _causal_adv20(pool, ohlcv)
    adv_dist = adv_distribution(adv, adv_threshold)

    adv_subset = pool[adv >= adv_threshold]
    delta_adv_subset = (layer1.delta_f_summary(adv_subset, h=h, min_rows_per_day=min_rows_per_day)
                        if len(adv_subset) else {"note": "ADV>=threshold部分集合が空"})

    return {
        "window": name,
        "n_warmup_rows_excluded": n_warmup_excluded,
        "n_rows_excluded_data_quality_9900T": int(n_excluded_dq_rows),
        "n_rows_after_exclusions": int(len(pool)),
        "n_days_total": n_days_total,
        "delta_f_all_days": delta_all,
        "delta_f_min_rows_filtered": delta_filtered,
        "excluded_days_below_min_rows": excluded_days,
        "adv20_jpy_distribution": adv_dist,
        "delta_f_adv_ge_threshold_subset": delta_adv_subset,
        "empty_files": empty_dates(replay_dir),
    }


def main(argv=None) -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long
    from stockbot.validation.layer1 import DEFAULT_H

    ap = argparse.ArgumentParser(prog="python scripts/robustness_liquidity_diagnostics.py")
    ap.add_argument("--main-replay-dir", default=None)
    ap.add_argument("--robustness-replay-dir", default=None)
    ap.add_argument("--h", type=int, default=DEFAULT_H)
    ap.add_argument("--min-rows-per-day", type=int, default=30)
    ap.add_argument("--adv-threshold", type=float, default=2e8)
    args = ap.parse_args(argv)

    if not args.main_replay_dir and not args.robustness_replay_dir:
        print("[robustness_liquidity_diagnostics] --main-replay-dir / --robustness-replay-dir "
              "のいずれも指定されていません")
        return 1

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    results = []
    if args.main_replay_dir:
        results.append(run_window("main", Path(args.main_replay_dir), ohlcv, args.h,
                                  args.min_rows_per_day, args.adv_threshold, exclude_warmup=False))
    if args.robustness_replay_dir:
        results.append(run_window("robustness", Path(args.robustness_replay_dir), ohlcv, args.h,
                                  args.min_rows_per_day, args.adv_threshold, exclude_warmup=True))

    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
