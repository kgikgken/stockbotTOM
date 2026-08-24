"""バックフィル進捗の分類レポート（診断用スクリプト。TASKS.md のタスクではない）。

data/store の実データ（backfill ワークフローが HISTORY_DAYS=2600 で取得したもの）を
上場一覧・流動性統計と突き合わせ、次の3分類に振り分ける:

  A: 取得成功、10年分あり（本数基準 int(HISTORY_DAYS*0.9) を達成）
  B: 取得成功だが履歴が短い。最初の観測日が 2016-08-01 より後 = 新規上場で
     そもそも10年分のデータが存在しない（不足ではなく正常）
  C: 取得失敗、または 2016-08-01 以前から存在するはずの銘柄で本数基準に
     届いていない（不自然な欠落の疑い。要調査）

市場区分・売買代金（ADV）帯ごとの内訳、B 分類の上場からの経過年数推定も出す。
スコアリング・本番パイプラインには一切影響しない読み取り専用の診断ツール。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockbot.data.store import OhlcvStore, from_long
from stockbot.universe.build import equities_only, liquidity_stats

HISTORY_DAYS = 2600
BAR_THRESHOLD = int(HISTORY_DAYS * 0.9)
NEW_LISTING_CUTOFF = pd.Timestamp("2016-08-01")


def adv_bucket(adv_jpy: float) -> str:
    if pd.isna(adv_jpy):
        return "不明(取得失敗)"
    if adv_jpy >= 2_000_000_000:
        return ">=20億円"
    if adv_jpy >= 200_000_000:
        return "2-20億円"
    if adv_jpy >= 50_000_000:
        return "0.5-2億円"
    return "<0.5億円"


def classify_row(row: pd.Series) -> str:
    if not row["fetched"]:
        return "C"
    if row["bars"] >= BAR_THRESHOLD:
        return "A"
    if pd.notna(row["first_date"]) and row["first_date"] > NEW_LISTING_CUTOFF:
        return "B"
    return "C"


def build_report(ohlcv: dict, listed: pd.DataFrame) -> pd.DataFrame:
    eq = equities_only(listed)[["ticker", "code", "name", "market", "sector33"]]
    stats = liquidity_stats(ohlcv)
    first_dates = {t: df.index.min() for t, df in ohlcv.items() if df is not None and len(df)}

    u = eq.merge(stats, on="ticker", how="left")
    u["first_date"] = u["ticker"].map(first_dates)
    u["fetched"] = u["bars"].notna()
    u["bars"] = u["bars"].fillna(0)
    u["class"] = u.apply(classify_row, axis=1)
    u["adv_bucket"] = u["adv_jpy"].apply(adv_bucket)
    today = pd.Timestamp.now().normalize()
    u["years_listed_est"] = (today - u["first_date"]).dt.days / 365.25
    return u


def main() -> int:
    store_dir = Path("data/store")
    daily_dir = Path("data/daily")
    ref_path = Path("data/reference/listed_latest.csv")

    store = OhlcvStore(store_dir, daily_dir)
    ohlcv = from_long(store.load())
    listed = pd.read_csv(ref_path, dtype={"code": str})

    u = build_report(ohlcv, listed)

    print(f"[coverage] 本数基準(>= {BAR_THRESHOLD}) / 新規上場境界 {NEW_LISTING_CUTOFF.date()}")
    print("=== 全体 ===")
    print(u["class"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int).to_string())
    print()
    print("=== 市場区分別 ===")
    print(pd.crosstab(u["market"], u["class"]).to_string())
    print()
    print("=== 売買代金帯別 ===")
    print(pd.crosstab(u["adv_bucket"], u["class"]).to_string())
    print()
    c = u[u["class"] == "C"].sort_values("bars")
    print(f"=== C 分類（要調査, n={len(c)}） ===")
    cols = ["ticker", "name", "market", "fetched", "bars", "first_date", "last_date", "adv_jpy"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(c[cols].to_string(index=False))
    print()
    b = u[u["class"] == "B"]
    print(f"=== B 分類（新規上場, n={len(b)}） 経過年数推定の分布 ===")
    print(b["years_listed_est"].describe().to_string())

    out_path = Path("backfill_coverage.csv")
    u.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n詳細を {out_path} に保存しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
