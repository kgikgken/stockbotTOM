"""33 業種の強弱（docs/SCREENER.md §2.9）。

順位そのものより、**候補の集合が変わらないこと**と**未来を見ないこと**を固定する。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.sector import (
    LONG_LOOKBACK,
    RANK_LOOKBACK,
    SECTOR_COLS,
    UNCLASSIFIED,
    rank_lookup,
    ranking_table,
    sector_strength,
)
from stockbot.screener.screen import NO_SECTOR_RANK, SECTOR_CAP, select_candidates

DATES = pd.bdate_range("2026-08-03", periods=40)
ASOF = DATES[30]


def frame(closes) -> pd.DataFrame:
    """終値だけ与えれば足りる（業種強弱は Close しか読まない）。"""
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c},
                        index=DATES[:len(c)])


def ramp(start: float, step: float, n: int = 40):
    return [start + step * i for i in range(n)]


class SectorStrengthTest(unittest.TestCase):
    def setUp(self):
        # 3業種。上げ幅の大きい順に 機械 > 銀行業 > 水産・農林業 になるよう作る
        self.ohlcv = {
            "1000.T": frame(ramp(100, 1.0)),    # 機械
            "1001.T": frame(ramp(200, 2.0)),    # 機械
            "2000.T": frame(ramp(100, 0.5)),    # 銀行業
            "3000.T": frame(ramp(100, -0.5)),   # 水産・農林業
        }
        self.sectors = {"1000.T": "機械", "1001.T": "機械",
                        "2000.T": "銀行業", "3000.T": "水産・農林業"}
        self.tickers = list(self.ohlcv)

    def _strength(self, **kw):
        return sector_strength(self.ohlcv, self.tickers, self.sectors, ASOF, **kw)

    def test_columns_and_rank_order(self):
        out = self._strength()
        self.assertEqual(list(out.columns), SECTOR_COLS)
        self.assertEqual(out["sector33"].tolist(), ["機械", "銀行業", "水産・農林業"])
        self.assertEqual(out["rank_5d"].tolist(), [1, 2, 3])
        self.assertEqual(out.loc[out["sector33"] == "機械", "n"].iloc[0], 2)

    def test_equal_weight_not_price_weight(self):
        """等加重。値がさ株に引っ張られない（時価総額も浮動株数も持っていないため）。

        安い方が +10%、高い方が 0% なら、業種のリターンは +5%。加重していたら
        高い方に寄って +5% にはならない。
        """
        cheap = [100.0] * 26 + [110.0] * 14      # T までに +10%
        rich = [10_000.0] * 40                   # 動かない
        ohlcv = {"c.T": frame(cheap), "r.T": frame(rich)}
        out = sector_strength(ohlcv, list(ohlcv),
                              {"c.T": "機械", "r.T": "機械"}, ASOF)
        self.assertAlmostEqual(float(out.loc[0, "ret_5d"]), 0.05, places=9)
        self.assertEqual(int(out.loc[0, "n"]), 2)

    def test_does_not_look_past_asof(self):
        """T より後の足を足しても結果が変わらない（CLAUDE.md 未来参照の禁止）。"""
        before = self._strength()
        bumped = {}
        for t, df in self.ohlcv.items():
            extra = df.copy()
            extra.iloc[31:] = extra.iloc[31:] * 10.0   # T より後だけ壊す
            bumped[t] = extra
        after = sector_strength(bumped, self.tickers, self.sectors, ASOF)
        pd.testing.assert_frame_equal(before, after)

    def test_ticker_without_asof_bar_is_excluded(self):
        """T の足が無い銘柄は業種の計算に入れない（前日で代用しない）。"""
        ohlcv = dict(self.ohlcv)
        ohlcv["1001.T"] = ohlcv["1001.T"].drop(index=ASOF)
        out = sector_strength(ohlcv, self.tickers, self.sectors, ASOF)
        self.assertEqual(int(out.loc[out["sector33"] == "機械", "n"].iloc[0]), 1)

    def test_short_history_drops_from_ranking(self):
        """5 日ぶん取れない銘柄は外れる。全滅した業種は表に出ない。"""
        ohlcv = dict(self.ohlcv)
        ohlcv["3000.T"] = frame(ramp(100, -0.5, n=3))
        out = sector_strength(ohlcv, self.tickers, self.sectors, ASOF)
        self.assertNotIn("水産・農林業", out["sector33"].tolist())
        self.assertEqual(out["rank_5d"].tolist(), [1, 2])

    def test_missing_sector_becomes_unclassified(self):
        out = sector_strength(self.ohlcv, self.tickers, {"1000.T": "機械"}, ASOF)
        self.assertIn(UNCLASSIFIED, out["sector33"].tolist())

    def test_ties_broken_by_sector_name(self):
        ohlcv = {"a.T": frame(ramp(100, 1.0)), "b.T": frame(ramp(100, 1.0))}
        out = sector_strength(ohlcv, list(ohlcv), {"a.T": "銀行業", "b.T": "機械"}, ASOF)
        self.assertEqual(out["sector33"].tolist(), ["機械", "銀行業"])   # 五十音ではなく文字列順

    def test_empty_input(self):
        out = sector_strength({}, [], {}, ASOF)
        self.assertEqual(list(out.columns), SECTOR_COLS)
        self.assertEqual(len(out), 0)
        self.assertEqual(rank_lookup(out), {})
        self.assertEqual(ranking_table(out), [])

    def test_lookbacks_are_the_documented_ones(self):
        self.assertEqual((RANK_LOOKBACK, LONG_LOOKBACK), (5, 20))

    def test_rank_lookup_and_ranking_table(self):
        out = self._strength()
        look = rank_lookup(out)
        self.assertEqual(look["機械"]["rank_5d"], 1)
        self.assertEqual(look["機械"]["n"], 2)
        table = ranking_table(out)
        self.assertEqual([r["sector33"] for r in table], ["機械", "銀行業", "水産・農林業"])
        self.assertEqual(table[0]["rank_5d"], 1)


class OrderingTest(unittest.TestCase):
    """並び順は変わるが、**候補の集合は変わらない**（§2.9）。"""

    def _passed(self, rows):
        return pd.DataFrame([{"ticker": t, "passes": True, "adv_jpy": adv}
                             for t, adv in rows])

    def setUp(self):
        # 銀行業は上限 3 件を超える 4 銘柄。機械は 1 銘柄
        self.df = self._passed([("B1.T", 90e8), ("B2.T", 80e8), ("B3.T", 70e8),
                                ("B4.T", 60e8), ("M1.T", 10e8)])
        self.sectors = {"B1.T": "銀行業", "B2.T": "銀行業", "B3.T": "銀行業",
                        "B4.T": "銀行業", "M1.T": "機械"}

    def test_candidate_set_is_identical_with_and_without_sector_rank(self):
        without = select_candidates(self.df, self.sectors)
        with_rank = select_candidates(self.df, self.sectors,
                                      sector_rank={"機械": {"rank_5d": 1},
                                                   "銀行業": {"rank_5d": 2}})
        self.assertEqual(set(without["ticker"]), set(with_rank["ticker"]))
        self.assertEqual(len(with_rank), SECTOR_CAP + 1)
        # 上限は業種ごとなので、同じ業種の中の選ばれ方（売買代金の降順）は変わらない
        self.assertEqual(sorted(without["ticker"]), sorted(with_rank["ticker"]))

    def test_order_follows_sector_rank_then_adv(self):
        out = select_candidates(self.df, self.sectors,
                                sector_rank={"機械": {"rank_5d": 1},
                                             "銀行業": {"rank_5d": 2}})
        self.assertEqual(out["ticker"].tolist(), ["M1.T", "B1.T", "B2.T", "B3.T"])

    def test_order_flips_when_sector_rank_flips(self):
        out = select_candidates(self.df, self.sectors,
                                sector_rank={"機械": {"rank_5d": 9},
                                             "銀行業": {"rank_5d": 1}})
        self.assertEqual(out["ticker"].tolist(), ["B1.T", "B2.T", "B3.T", "M1.T"])

    def test_without_ranking_it_is_adv_descending(self):
        out = select_candidates(self.df, self.sectors, sector_rank={})
        self.assertEqual(out["ticker"].tolist(), ["B1.T", "B2.T", "B3.T", "M1.T"])
        self.assertTrue((out["sector_rank_5d"] == NO_SECTOR_RANK).all())

    def test_sector_without_rank_goes_last(self):
        out = select_candidates(self.df, self.sectors,
                                sector_rank={"銀行業": {"rank_5d": 5}})
        self.assertEqual(out["ticker"].tolist()[-1], "M1.T")

    def test_empty_input_keeps_columns(self):
        out = select_candidates(self.df.iloc[0:0], self.sectors)
        self.assertIn("sector33", out.columns)
        self.assertIn("sector_rank_5d", out.columns)


if __name__ == "__main__":
    unittest.main()
