"""基礎指標（T-201）。手計算との一致と、再計算一致テスト（DESIGN.md §11）。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.indicators import (
    atr_simple,
    atr_wilder,
    bb_width,
    sma,
    true_range,
    weekly_ohlcv,
)


def _ohlc(highs, lows, closes, start="2026-01-05"):
    idx = pd.bdate_range(start=start, periods=len(highs))
    return (pd.Series(highs, index=idx, dtype=float),
            pd.Series(lows, index=idx, dtype=float),
            pd.Series(closes, index=idx, dtype=float))


class SmaTest(unittest.TestCase):
    def test_matches_hand_calc(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = sma(close, 3)
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
        for got, want in zip(out.tolist(), expected):
            if np.isnan(want):
                self.assertTrue(np.isnan(got))
            else:
                self.assertAlmostEqual(got, want, places=9)


class TrueRangeTest(unittest.TestCase):
    def test_matches_hand_calc_and_first_row_nan(self):
        # H, L, C
        high, low, close = _ohlc([10, 12, 11, 15], [8, 9, 9, 10], [9, 11, 10, 14])
        tr = true_range(high, low, close)
        self.assertTrue(np.isnan(tr.iloc[0]))
        # day2: H-L=3, |H-Cprev|=|12-9|=3, |L-Cprev|=|9-9|=0 -> max=3
        self.assertAlmostEqual(tr.iloc[1], 3.0, places=9)
        # day3: H-L=2, |H-Cprev|=|11-11|=0, |L-Cprev|=|9-11|=2 -> max=2
        self.assertAlmostEqual(tr.iloc[2], 2.0, places=9)
        # day4: H-L=5, |H-Cprev|=|15-10|=5, |L-Cprev|=|10-10|=0 -> max=5
        self.assertAlmostEqual(tr.iloc[3], 5.0, places=9)


class AtrWilderTest(unittest.TestCase):
    def test_matches_independent_hand_calc(self):
        # 独立に（テストコード側で）Wilder 法を再実装し、モジュール実装と突き合わせる。
        rng = np.random.default_rng(1)
        n = 20
        high = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)) + rng.uniform(0.5, 2, n))
        low = high - rng.uniform(0.5, 2, n)
        close = low + rng.uniform(0, 1, n) * (high - low)
        idx = pd.bdate_range("2026-01-05", periods=n)
        high.index = low.index = close.index = idx

        period = 5
        got = atr_wilder(high, low, close, n=period)

        # 独立実装（テストコード側）
        tr = [np.nan]
        for i in range(1, n):
            tr.append(max(high.iloc[i] - low.iloc[i],
                          abs(high.iloc[i] - close.iloc[i - 1]),
                          abs(low.iloc[i] - close.iloc[i - 1])))
        expected = [np.nan] * n
        seed_idx = period  # tr[0] は NaN なので有効TRは tr[1..n-1]、seed は tr[1..period]の平均 -> index=period
        expected[seed_idx] = float(np.mean(tr[1:1 + period]))
        for i in range(seed_idx + 1, n):
            expected[i] = (expected[i - 1] * (period - 1) + tr[i]) / period

        for i in range(n):
            if np.isnan(expected[i]):
                self.assertTrue(np.isnan(got.iloc[i]), f"index {i} should be NaN")
            else:
                self.assertAlmostEqual(got.iloc[i], expected[i], places=9, msg=f"index {i}")

    def test_simple_known_series(self):
        # H-L のみで TR が決まる単純系列（C_prev との差が常に小さくなるよう close を H,L の中間に固定）
        high, low, close = _ohlc([10, 10, 10, 10, 10], [8, 8, 8, 8, 8], [9, 9, 9, 9, 9])
        got = atr_wilder(high, low, close, n=3)
        # tr = [NaN, 2,2,2,2]（H-L=2, |H-Cprev|=1, |L-Cprev|=1 -> max=2）
        # seed(index=3) = mean(tr[1:4]) = 2.0 ; 以降も 2.0 のまま
        self.assertTrue(np.isnan(got.iloc[0]))
        self.assertTrue(np.isnan(got.iloc[1]))
        self.assertTrue(np.isnan(got.iloc[2]))
        self.assertAlmostEqual(got.iloc[3], 2.0, places=9)
        self.assertAlmostEqual(got.iloc[4], 2.0, places=9)


class AtrSimpleTest(unittest.TestCase):
    def test_matches_rolling_mean_of_tr(self):
        high, low, close = _ohlc([10, 12, 11, 15, 14], [8, 9, 9, 10, 12], [9, 11, 10, 14, 13])
        tr = true_range(high, low, close)
        got = atr_simple(high, low, close, n=2)
        expected = tr.rolling(2, min_periods=2).mean()
        pd.testing.assert_series_equal(got, expected)


class BbWidthTest(unittest.TestCase):
    def test_matches_hand_calc_population_std(self):
        close = pd.Series([10.0, 11.0, 9.0, 12.0, 8.0])
        got = bb_width(close, n=5, k=2.0)
        mean = close.mean()
        std = close.std(ddof=0)
        expected_last = (2 * 2.0 * std) / mean
        self.assertTrue(np.isnan(got.iloc[3]))
        self.assertAlmostEqual(got.iloc[4], expected_last, places=9)


class WeeklyOhlcvTest(unittest.TestCase):
    def _daily(self, n_weeks=3, start="2026-07-27"):
        # 2026-07-27 は月曜。3週間ぶんの平日を作る（各週 Mon〜Fri）
        idx = pd.bdate_range(start=start, periods=n_weeks * 5)
        n = len(idx)
        close = pd.Series(np.arange(n) + 100.0, index=idx)
        df = pd.DataFrame({
            "Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0,
            "Close": close, "Volume": np.arange(n) + 1.0,
        }, index=idx)
        return df

    def test_aggregation_matches_hand_calc_for_confirmed_week(self):
        df = self._daily(n_weeks=3)
        # asof を第3週の水曜にして、第1週・第2週が確定週として出るはず
        asof = df.index[11]  # 3週目の Wed (0-indexed週: week0=0-4,week1=5-9,week2=10-14)
        w = weekly_ohlcv(df, asof)
        self.assertEqual(len(w), 2)
        wk1 = df.iloc[0:5]
        self.assertAlmostEqual(w.iloc[0]["Open"], wk1["Open"].iloc[0])
        self.assertAlmostEqual(w.iloc[0]["High"], wk1["High"].max())
        self.assertAlmostEqual(w.iloc[0]["Low"], wk1["Low"].min())
        self.assertAlmostEqual(w.iloc[0]["Close"], wk1["Close"].iloc[-1])
        self.assertAlmostEqual(w.iloc[0]["Volume"], wk1["Volume"].sum())

    def test_excludes_week_containing_asof_even_on_friday(self):
        df = self._daily(n_weeks=2)
        friday = df.index[4]  # 第1週の金曜
        w = weekly_ohlcv(df, friday)
        # 金曜であっても、その週は「未確定」として除外される
        self.assertEqual(len(w), 0)

    def test_future_rows_within_asof_week_do_not_leak_into_output(self):
        # T の週の未来分（T より後の日）が daily に含まれていても、
        # その週自体が丸ごと除外されるため出力に影響しないこと。
        df = self._daily(n_weeks=2)
        asof = df.index[6]  # 第2週の火曜（第2週内に asof より後の日が存在する）
        w_partial = weekly_ohlcv(df.loc[:asof], asof)
        w_full = weekly_ohlcv(df, asof)
        pd.testing.assert_frame_equal(w_partial, w_full)
        self.assertEqual(len(w_full), 1)  # 第1週のみ確定


class RecalcConsistencyTest(unittest.TestCase):
    """DESIGN.md §11: T で切って計算 == 全期間で計算して T 行を取り出す。"""

    def setUp(self):
        rng = np.random.default_rng(7)
        n = 260
        idx = pd.bdate_range("2025-06-02", periods=n)
        close = pd.Series(1000 + np.cumsum(rng.normal(0, 5, n)), index=idx)
        high = close + rng.uniform(1, 5, n)
        low = close - rng.uniform(1, 5, n)
        open_ = low + rng.uniform(0, 1, n) * (high - low)
        vol = rng.uniform(1e4, 1e6, n)
        self.df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                                "Close": close, "Volume": vol}, index=idx)
        self.t_points = [30, 60, 120, 200, 259]

    def test_sma_all_periods(self):
        for n in (5, 25, 75, 200):
            full = sma(self.df["Close"], n)
            for t in self.t_points:
                cut = sma(self.df["Close"].iloc[:t + 1], n)
                got, want = cut.iloc[-1], full.iloc[t]
                if np.isnan(want):
                    self.assertTrue(np.isnan(got))
                else:
                    self.assertAlmostEqual(got, want, places=9)

    def test_true_range(self):
        full = true_range(self.df["High"], self.df["Low"], self.df["Close"])
        for t in self.t_points:
            sub = self.df.iloc[:t + 1]
            cut = true_range(sub["High"], sub["Low"], sub["Close"])
            self.assertAlmostEqual(cut.iloc[-1], full.iloc[t], places=9)

    def test_atr_wilder(self):
        full = atr_wilder(self.df["High"], self.df["Low"], self.df["Close"], n=14)
        for t in self.t_points:
            sub = self.df.iloc[:t + 1]
            cut = atr_wilder(sub["High"], sub["Low"], sub["Close"], n=14)
            self.assertAlmostEqual(cut.iloc[-1], full.iloc[t], places=9)

    def test_atr_simple_5_and_20(self):
        for n in (5, 20):
            full = atr_simple(self.df["High"], self.df["Low"], self.df["Close"], n)
            for t in self.t_points:
                sub = self.df.iloc[:t + 1]
                cut = atr_simple(sub["High"], sub["Low"], sub["Close"], n)
                self.assertAlmostEqual(cut.iloc[-1], full.iloc[t], places=9)

    def test_bb_width(self):
        full = bb_width(self.df["Close"])
        for t in self.t_points:
            cut = bb_width(self.df["Close"].iloc[:t + 1])
            self.assertAlmostEqual(cut.iloc[-1], full.iloc[t], places=9)

    def test_weekly_ohlcv(self):
        for t in self.t_points:
            asof = self.df.index[t]
            full = weekly_ohlcv(self.df, asof)
            cut = weekly_ohlcv(self.df.iloc[:t + 1], asof)
            pd.testing.assert_frame_equal(full, cut)
            # T を含む週が含まれていないこと
            self.assertNotIn(asof, full.index)
            for wk_end in full.index:
                self.assertLess(wk_end, asof if asof.weekday() == 4 else
                                asof + pd.offsets.Week(weekday=4))


if __name__ == "__main__":
    unittest.main()
