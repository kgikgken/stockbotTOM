"""設計窓(a′)・頑健性窓のreplay実行が、期待される営業日数ぶん過不足なく保存
されているかを突合する（診断専用。TASKS.md のタスクではない。2026-08-29 指示:
GitHub Actions のキャッシュ上書き等で静かに欠けた日がΔF_tの時系列に穴を
空けていないかを確認する）。

期待営業日数は validation.replay._real_trading_days と同じデータ駆動の
非取引日判定（store内の全銘柄のうちその日にデータを持つ銘柄の割合が
MIN_TRADING_DAY_COVERAGE 未満の日を除く）で、pd.bdate_range(start, end) から
実際の取引日だけを残して算出する。実際に保存されたファイル数・欠けている日付・
余剰の日付（非取引日と判定されたのに保存されている＝逆側の不整合）を報告する。

store・data/a_prime・data/replay_robustness のいずれも変更しない読み取り専用。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd


def saved_dates(replay_dir: Path, prefix: str) -> set:
    out = set()
    replay_dir = Path(replay_dir)
    if not replay_dir.exists():
        return out
    for f in replay_dir.glob(f"{prefix}_*.csv.gz"):
        date_str = f.name[len(prefix) + 1: -len(".csv.gz")]
        try:
            out.add(pd.Timestamp(date_str))
        except ValueError:
            continue
    return out


def check_coverage(start: pd.Timestamp, end: pd.Timestamp, replay_dir: Path, prefix: str,
                   ohlcv: Dict[str, pd.DataFrame], log=print) -> dict:
    from stockbot.validation.replay import _real_trading_days

    expected_dates = pd.bdate_range(start, end)
    expected_dates = _real_trading_days(expected_dates, ohlcv, log=lambda *a: None)
    expected = set(expected_dates)
    saved = saved_dates(replay_dir, prefix)
    saved_in_range = {d for d in saved if start <= d <= end}

    missing = sorted(expected - saved_in_range)
    extra = sorted(saved_in_range - expected)
    result = {
        "n_expected": len(expected), "n_saved_in_range": len(saved_in_range),
        "n_missing": len(missing), "n_extra": len(extra),
        "missing": [d.date().isoformat() for d in missing],
        "extra": [d.date().isoformat() for d in extra],
    }
    log(f"[coverage] {replay_dir}: 期待営業日数={result['n_expected']} / "
        f"保存日数(範囲内)={result['n_saved_in_range']} / 不足={result['n_missing']} / "
        f"余剰={result['n_extra']}")
    if missing:
        log(f"[coverage] {replay_dir}: 欠けている日付: {result['missing']}")
    if extra:
        log(f"[coverage] {replay_dir}: 余剰の日付: {result['extra']}")
    return result


def main(argv=None) -> int:
    from stockbot.config import Settings
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    print("=== 設計窓(a′) 2021-08-01〜2024-01-31 ===")
    check_coverage(pd.Timestamp("2021-08-01"), pd.Timestamp("2024-01-31"),
                   Path("data/a_prime"), "a_prime", ohlcv)

    print("\n=== 頑健性窓 2017-02-13〜2021-07-31（g3_start基準、warmup含む） ===")
    check_coverage(pd.Timestamp("2017-02-13"), pd.Timestamp("2021-07-31"),
                   Path("data/replay_robustness"), "replay", ohlcv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
