"""data/store/revisions.csv.gz の記録範囲を報告する（診断専用。TASKS.mdのタスクでは
ない）。「全履歴走査で9900.T以外に該当なし」という結果の実際の意味（何年分を
実際にカバーしているか）を確認するため、全銘柄を通じた最古のobserved_on・dateを
出す。store は一切変更しない読み取り専用。
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> int:
    from stockbot.config import Settings
    from stockbot.data.store import OhlcvStore

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    if not store.revisions_path.exists():
        print("[revisions-coverage] revisions.csv.gz が無い（改訂履歴なし）")
        return 0

    rev = pd.read_csv(store.revisions_path)
    if len(rev) == 0:
        print("[revisions-coverage] revisions.csv.gz は空")
        return 0

    rev["observed_on"] = pd.to_datetime(rev["observed_on"], format="mixed")
    rev["date"] = pd.to_datetime(rev["date"], format="mixed")

    print(f"[revisions-coverage] 総改訂件数={len(rev)} "
          f"銘柄数={rev['ticker'].nunique()}")
    print(f"[revisions-coverage] observed_on（改訂が記録された日）の範囲: "
          f"{rev['observed_on'].min().date()} 〜 {rev['observed_on'].max().date()}")
    print(f"[revisions-coverage] date（改訂対象の取引日）の範囲: "
          f"{rev['date'].min().date()} 〜 {rev['date'].max().date()}")

    by_observed = rev.groupby(rev["observed_on"].dt.date).size().sort_index()
    print("[revisions-coverage] observed_on別の件数（全件）:")
    for d, n in by_observed.items():
        print(f"  {d}: {n}件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
