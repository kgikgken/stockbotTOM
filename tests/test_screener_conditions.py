"""19 条件のブール判定（docs/SCREENER.md §2）。

全条件が通る合成系列を 1 本用意し、条件ごとに 1 つだけ壊して「その条件が落ちる」ことを
確かめる。欠損時は不成立（A4 のみ例外）も併せて確認する。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.dimensions import compute_dimensions
from stockbot.features.indicators import atr_wilder, sma
from stockbot.features.pullback import pullback_state
from stockbot.features.swings import alternate_swings, detect_raw_swings
from stockbot.screener.conditions import (
    CONDITION_IDS,
    GROUPS,
    SELF_CONTAINED_IDS,
    average_turnover,
    evaluate_conditions,
    passes_self_contained,
    rs60,
)
from stockbot.universe.build import liquidity_stats

K = 3
H0_POS = 264
LP_POS = 271


def base_closes():
    """全条件が通る形の終値。warmup → L0のV字 → 上昇 → H0のピーク → 押し目と反発。"""
    warmup = np.linspace(300.0, 540.0, 240)
    dip = np.array([540, 536, 532, 528, 532, 536, 540.0])
    rise = np.linspace(540.0, 650.0, 15)[1:]
    peak = np.array([650, 654, 657, 660, 657, 654, 650.0])
    tail = np.array([640, 630, 625, 622, 632, 642.0])
    return np.concatenate([warmup, dip, rise, peak, tail])


def make_frame(closes=None, volume=1e6):
    closes = base_closes() if closes is None else np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2023-01-02", periods=len(closes))
    close = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 0.3,
        "Low": close - 0.3,
        "Close": close,
        "Volume": pd.Series(np.full(len(closes), float(volume)), index=idx),
        "Dividends": pd.Series(np.zeros(len(closes)), index=idx),
        "Stock Splits": pd.Series(np.zeros(len(closes)), index=idx),
    }, index=idx)


def make_index(n, drift=0.0002, start=2000.0):
    """指数。既定では銘柄より緩やかに上げるので rs60 は正になる。"""
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(start * np.exp(np.arange(n) * drift), index=idx)


def evaluate(df=None, idx_close=None, earnings_schedule=None, t_pos=None):
    df = make_frame() if df is None else df
    t_pos = len(df) - 1 if t_pos is None else t_pos
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    alt = alternate_swings(detect_raw_swings(high, low, K))
    pb = pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                        atr_wilder(high, low, close, 14), alt, t_pos, K)
    if idx_close is None:
        idx_close = make_index(len(df))
    return evaluate_conditions("1234.T", high, low, close, volume, pb, t_pos,
                               idx_close=idx_close.reindex(close.index).ffill(limit=3),
                               earnings_schedule=earnings_schedule), pb


class BaselinePassesTest(unittest.TestCase):
    """基準の系列は E1 以外の 18 条件を全て満たす。ここが崩れると他のテストが無意味になる。"""

    @classmethod
    def setUpClass(cls):
        cls.row, cls.pb = evaluate()

    def test_all_eighteen_pass(self):
        failed = [cid for cid in SELF_CONTAINED_IDS if not self.row[cid]]
        self.assertEqual(failed, [], f"落ちた条件: {failed}")
        self.assertTrue(passes_self_contained(self.row))

    def test_e1_is_pending(self):
        self.assertTrue(pd.isna(self.row["E1"]))

    def test_structure_is_where_expected(self):
        self.assertEqual(self.pb["h0"], H0_POS)
        self.assertEqual(self.pb["lp"], LP_POS)

    def test_landing_ma_recorded(self):
        self.assertIn(self.row["landing_ma"], ("SMA5", "SMA25", "SMA75"))
        self.assertLessEqual(self.row["landing_dist_atr"], 1.0)

    def test_condition_ids_are_nineteen(self):
        self.assertEqual(len(CONDITION_IDS), 19)
        self.assertEqual(len(SELF_CONTAINED_IDS), 18)
        self.assertEqual([len(v) for v in GROUPS.values()], [4, 4, 4, 4, 3])


class ConditionBreaksTest(unittest.TestCase):
    """条件ごとに 1 つだけ壊す。"""

    def assert_fails(self, cid, row):
        self.assertFalse(bool(row[cid]), f"{cid} が落ちていない")

    def test_a1_low_turnover(self):
        row, _pb = evaluate(make_frame(volume=1e2))     # 売買代金 6.4万円
        self.assert_fails("A1", row)

    def test_a2_low_price(self):
        row, _pb = evaluate(make_frame(base_closes() / 3.0))   # 終値 214円
        self.assert_fails("A2", row)
        self.assertTrue(row["A1"] is False or row["A2"] is False)

    def test_a3_short_history(self):
        short = make_frame().iloc[-200:]
        row, _pb = evaluate(short)
        self.assert_fails("A3", row)
        self.assert_fails("B4", row)   # 252本に満たないので B4 も判定できない

    def test_a4_earnings_within_five_days(self):
        df = make_frame()
        schedule = pd.DataFrame({"ticker": ["1234.T"],
                                 "date": [df.index[-1] + pd.Timedelta(days=3)]})
        row, _pb = evaluate(df, earnings_schedule=schedule)
        self.assert_fails("A4", row)
        self.assertFalse(bool(row["a4_earnings_unknown"]))

    def test_a4_far_earnings_passes(self):
        df = make_frame()
        schedule = pd.DataFrame({"ticker": ["1234.T"],
                                 "date": [df.index[-1] + pd.Timedelta(days=40)]})
        row, _pb = evaluate(df, earnings_schedule=schedule)
        self.assertTrue(bool(row["A4"]))
        self.assertFalse(bool(row["a4_earnings_unknown"]))

    def test_a4_unknown_passes_and_is_flagged(self):
        """決算日が取れないときだけ成立扱い（§2.3 唯一の例外）。"""
        row, _pb = evaluate(earnings_schedule=pd.DataFrame({"ticker": ["9999.T"],
                                                            "date": [pd.Timestamp("2030-01-01")]}))
        self.assertTrue(bool(row["A4"]))
        self.assertTrue(bool(row["a4_earnings_unknown"]))

    def test_b_group_fails_on_downtrend(self):
        closes = base_closes()[::-1].copy()   # 下降トレンドに反転
        row, _pb = evaluate(make_frame(closes))
        for cid in ("B1", "B2"):
            self.assert_fails(cid, row)

    def test_b3_below_sma200(self):
        closes = base_closes().copy()
        closes[-1] = 380.0                    # SMA200 を割る終値
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("B3", row)

    def test_c1_too_shallow(self):
        closes = base_closes().copy()
        closes[268:] = [655, 653, 651, 650, 653, 656]   # depth_pct < 3%
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("C1", row)

    def test_c1_too_deep(self):
        closes = base_closes().copy()
        closes[268:] = [630, 610, 590, 580, 595, 610]   # depth_pct > 8%
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("C1", row)

    def test_c3_too_long(self):
        """押し目が 12 日を超える（tail を伸ばす）。"""
        closes = np.concatenate([base_closes()[:268],
                                 [640, 636, 632, 628, 625, 623, 622, 624, 628, 632, 636, 642.0]])
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("C3", row)

    def test_d1_d2_fail_without_bounce(self):
        closes = base_closes().copy()
        closes[-1] = 623.0     # 5日線の下・前日高値の下で引ける
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("D1", row)
        self.assert_fails("D2", row)

    def test_d3_position_too_low(self):
        closes = base_closes().copy()
        closes[-2:] = [624.0, 627.0]   # Lp からほとんど戻っていない
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("D3", row)

    def test_e2_dev5_too_hot(self):
        closes = base_closes().copy()
        closes[-1] = 660.0 - 0.5       # 5日線から大きく上に伸びる（H0 は超えない）
        row, _pb = evaluate(make_frame(closes))
        self.assert_fails("E2", row)

    def test_e3_lp_too_old(self):
        """押し安値が 5 営業日より前（反発が長く続いている）。"""
        closes = np.concatenate([base_closes()[:272],
                                 [628, 632, 634, 636, 638, 642.0]])
        row, _pb = evaluate(make_frame(closes))
        self.assertGreater(row["lp_age"], 5)
        self.assert_fails("E3", row)

    def test_missing_structure_fails_c_and_d_and_e(self):
        """押し目構造が無い（単調増加だけ）と C・D3・D4・E2・E3 が落ちる。"""
        closes = np.linspace(300.0, 700.0, 300)
        row, pb = evaluate(make_frame(closes))
        self.assertIsNone(pb["h0"])
        for cid in ("C1", "C2", "C3", "C4", "D3", "D4", "E2", "E3"):
            self.assert_fails(cid, row)


class HelperTest(unittest.TestCase):
    def test_average_turnover_matches_liquidity_stats(self):
        df = make_frame()
        mine = average_turnover(df["Close"], df["Volume"], len(df) - 1)
        theirs = float(liquidity_stats({"1234.T": df}, 20)["adv_jpy"].iloc[0])
        self.assertAlmostEqual(mine, theirs, places=4)

    def test_average_turnover_needs_full_window(self):
        df = make_frame()
        self.assertTrue(np.isnan(average_turnover(df["Close"], df["Volume"], 5)))

    def test_rs60_matches_d2_rs60(self):
        """rs60 は DESIGN.md §5 D2 の d2_rs60 と同じ値になる。"""
        df = make_frame()
        t_pos = len(df) - 1
        idx_close = make_index(len(df)).reindex(df.index).ffill(limit=3)
        high, low, close = df["High"], df["Low"], df["Close"]
        alt = alternate_swings(detect_raw_swings(high, low, K))
        pb = pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                            atr_wilder(high, low, close, 14), alt, t_pos, K)
        feats, _extra = compute_dimensions(df["Open"], high, low, close, df["Volume"],
                                           df["Dividends"], alt, pb, t_pos, K,
                                           idx_close=idx_close)
        expected = float(feats.loc[feats["id"] == "d2_rs60", "value"].iloc[0])
        self.assertAlmostEqual(rs60(close, idx_close, t_pos), expected, places=10)

    def test_rs60_missing_index_is_nan(self):
        df = make_frame()
        self.assertTrue(np.isnan(rs60(df["Close"], None, len(df) - 1)))


if __name__ == "__main__":
    unittest.main()
