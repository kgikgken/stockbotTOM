import unittest

import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.jpx_lists import norm_ticker, normalize_listed
from stockbot.data.synthetic import make_synthetic, synthetic_listed
from stockbot.universe.build import build_universe, liquidity_stats


class TickerTest(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(norm_ticker("1301"), "1301.T")
        self.assertEqual(norm_ticker("130A"), "130A.T")
        self.assertEqual(norm_ticker("8035T"), "8035.T")
        self.assertEqual(norm_ticker("7203.T"), "7203.T")
        self.assertEqual(norm_ticker("25935"), "25935")   # 優先株はそのまま（除外対象）


class ListedTest(unittest.TestCase):
    def test_jpx_columns_and_exclusions(self):
        raw = pd.DataFrame({
            "日付": ["20260731"] * 5,
            "コード": ["1301", "1305", "3283", "25935", "130A"],
            "銘柄名": ["極洋", "ETF", "REIT", "優先株", "新コード"],
            "市場・商品区分": ["プライム（内国株式）", "ETF・ETN",
                           "REIT・ベンチャーファンド・カントリーファンド・インフラファンド",
                           "プライム（内国株式）", "グロース（内国株式）"],
            "33業種コード": [""] * 5, "33業種区分": ["水産・農林業", "", "", "食料品", "医薬品"],
            "17業種コード": [""] * 5, "17業種区分": [""] * 5, "規模コード": [""] * 5, "規模区分": [""] * 5,
        })
        df = normalize_listed(raw)
        eq = df[df["is_equity"]]["ticker"].tolist()
        self.assertListEqual(eq, ["1301.T", "130A.T"])

    def test_seed_csv_columns(self):
        raw = pd.DataFrame({"ticker": ["1301.T"], "name": ["極洋"], "sector": ["水産・農林業"],
                            "industry_big": ["食品"], "market": ["Prime"], "earnings_date": [""]})
        df = normalize_listed(raw)
        self.assertTrue(bool(df["is_equity"].iloc[0]))
        self.assertEqual(df["sector33"].iloc[0], "水産・農林業")


class UniverseTest(unittest.TestCase):
    def test_filters(self):
        listed = synthetic_listed(10)
        ohlcv = make_synthetic(listed["ticker"].tolist(), n_bars=300, seed=1)
        # 1銘柄を履歴不足に、1銘柄を出来高ゼロに
        t0, t1 = listed["ticker"].iloc[0], listed["ticker"].iloc[1]
        ohlcv[t0] = ohlcv[t0].tail(100)
        ohlcv[t1] = ohlcv[t1].assign(Volume=0.0)
        stats = liquidity_stats(ohlcv, 20)
        u = build_universe(listed, stats, min_adv_jpy=1.0, min_price=1.0, min_history_bars=250)
        row0 = u[u["ticker"] == t0].iloc[0]
        row1 = u[u["ticker"] == t1].iloc[0]
        self.assertFalse(bool(row0["passes"]))
        self.assertFalse(bool(row0["ok_history"]))
        self.assertFalse(bool(row1["ok_adv"]))
        self.assertEqual(int(u["passes"].sum()), 8)


if __name__ == "__main__":
    unittest.main()
