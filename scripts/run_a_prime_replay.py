"""(a′) 母集団（DESIGN.md §10.2-4: G0〜G3を通過しているが押し目状態でない銘柄）の
日次超過リターンを計算する（T-403 / §9.3 の h 選択に使う）。

validation.replay.run_a_prime_replay の薄いCLIラッパー。中断再開は既存の
run_replay と同じ規約（既にその日のファイルがあればスキップ）。store・
data/replay のいずれも変更しない（別ディレクトリ data/a_prime 等に出力する）。

使い方: python scripts/run_a_prime_replay.py --start 2021-08-01 --end 2024-01-31
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.jpx_lists import load_earnings_schedule, normalize_listed
    from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long
    from stockbot.validation.replay import run_a_prime_replay

    ap = argparse.ArgumentParser(prog="python scripts/run_a_prime_replay.py")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--output-dir", default=None, help="既定: <DATA_DIR>/a_prime")
    ap.add_argument("--include-holdout", action="store_true",
                    help="ホールドアウト期間も生成する（通常は指定しない。CLAUDE.md 絶対規則）")
    args = ap.parse_args(argv)

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    ohlcv.pop(IDX_TICKER, None)

    listed_path = cfg.reference_dir / "listed_latest.csv"
    if not listed_path.exists():
        print("[a_prime] listed_latest.csv が無いため中止（先に cli.py listed を実行）")
        return 1
    listed = normalize_listed(pd.read_csv(listed_path, dtype=str).fillna(""))

    earnings_schedule = None
    p = cfg.reference_dir / "earnings_schedule.csv"
    if p.exists():
        earnings_schedule = load_earnings_schedule(p)

    output_dir = Path(args.output_dir) if args.output_dir else cfg.data_dir / "a_prime"
    run_a_prime_replay(ohlcv, listed, output_dir,
                       pd.Timestamp(args.start), pd.Timestamp(args.end),
                       cfg.k, cfg.label_n, cfg.min_history_bars,
                       earnings_schedule=earnings_schedule, include_holdout=args.include_holdout,
                       log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
