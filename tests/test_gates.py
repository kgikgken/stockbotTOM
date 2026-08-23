"""ゲート G0〜G3（T-208）。各ゲートが単独で効くこと・再計算一致・決算欠測時の扱い。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.gates import evaluate_gates
from stockbot.features.indicators import sma

N = 40
T = N - 1


def _series(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, dtype=float, index=idx)


def _base_arrays():
    """全ゲートが素直に通る手作りの配列（close, high, sma75, sma200）。"""
    close = np.linspace(80.0, 105.0, N)
    high = close.copy()
    sma75 = np.full(N, np.nan)
    sma75[T] = 110.0
    sma200 = np.full(N, np.nan)
    sma200[T] = 100.0
    sma200[T - 30] = 90.0
    return close, high, sma75, sma200


class GateIndependenceTest(unittest.TestCase):
    """各ゲートが単独で（他を満たしたまま）落ちること。"""

    def test_all_pass_baseline(self):
        close, high, sma75, sma200 = _base_arrays()
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertTrue(r["gate_pass"])
        self.assertTrue(r["g0"] and r["g1"] and r["g2"] and r["g3"])

    def test_g0_fails_alone_when_universe_fails(self):
        close, high, sma75, sma200 = _base_arrays()
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=False, label_n=15)
        self.assertFalse(r["g0"])
        self.assertFalse(r["gate_pass"])
        self.assertTrue(r["g1"] and r["g2"] and r["g3"])

    def test_g0_fails_alone_when_earnings_near(self):
        close, high, sma75, sma200 = _base_arrays()
        close_s = _series(close)
        asof = close_s.index[T]
        schedule = pd.DataFrame({
            "ticker": ["1234.T"],
            "date": [asof + pd.Timedelta(days=5)],
        })
        r = evaluate_gates(close_s, _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15,
                           earnings_schedule=schedule, ticker="1234.T")
        self.assertFalse(r["g0"])
        self.assertFalse(r["g0_earnings_unknown"])
        self.assertFalse(r["gate_pass"])
        self.assertTrue(r["g1"] and r["g2"] and r["g3"])

    def test_g0_passes_when_earnings_far(self):
        close, high, sma75, sma200 = _base_arrays()
        close_s = _series(close)
        asof = close_s.index[T]
        schedule = pd.DataFrame({
            "ticker": ["1234.T"],
            "date": [asof + pd.Timedelta(days=60)],
        })
        r = evaluate_gates(close_s, _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15,
                           earnings_schedule=schedule, ticker="1234.T")
        self.assertTrue(r["g0"])
        self.assertFalse(r["g0_earnings_unknown"])

    def test_g0_passes_with_unknown_flag_when_no_schedule(self):
        close, high, sma75, sma200 = _base_arrays()
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertTrue(r["g0"])
        self.assertTrue(r["g0_earnings_unknown"])

    def test_g1_fails_alone_when_sma75_below_sma200(self):
        close, high, sma75, sma200 = _base_arrays()
        sma75[T] = 95.0  # < sma200[T]=100
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertFalse(r["g1"])
        self.assertFalse(r["gate_pass"])
        self.assertTrue(r["g0"] and r["g2"] and r["g3"])

    def test_g1_fails_alone_when_sma200_not_rising(self):
        close, high, sma75, sma200 = _base_arrays()
        sma200[T - 30] = 105.0  # sma200[T]=100 <= sma200[T-30] -> 下向き
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertFalse(r["g1"])
        self.assertTrue(r["g0"] and r["g2"] and r["g3"])

    def test_g1_false_when_sma200_history_insufficient(self):
        close, high, sma75, sma200 = _base_arrays()
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           5, passes_universe=True, label_n=15)  # t_pos-30 < 0
        self.assertFalse(r["g1"])

    def test_g2_fails_alone_when_close_below_sma200(self):
        close, high, sma75, sma200 = _base_arrays()
        close[T] = 95.0  # < sma200[T]=100
        high[T] = 95.0
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertFalse(r["g2"])
        self.assertTrue(r["g0"] and r["g1"] and r["g3"])

    def test_g3_fails_alone_when_far_below_52w_high(self):
        close, high, sma75, sma200 = _base_arrays()
        high[20] = 200.0  # 過去に52週高値を作る
        close[T] = 100.0  # 100 < 0.75*200=150
        sma200[T] = 90.0  # G2(close>=sma200)は維持
        sma200[T - 30] = 80.0  # g1: sma200[T](90) > sma200[T-30](80) を維持
        sma75[T] = 96.0  # g1: sma75>sma200(90) を維持
        r = evaluate_gates(_series(close), _series(high), _series(sma75), _series(sma200),
                           T, passes_universe=True, label_n=15)
        self.assertFalse(r["g3"])
        self.assertTrue(r["g0"] and r["g1"] and r["g2"], msg=r)


class RecalcConsistencyTest(unittest.TestCase):
    """DESIGN.md §11: 実際の指標計算(sma)を通した再計算一致。"""

    def test_truncated_matches_full(self):
        rng = np.random.default_rng(4)
        n = 320
        close = _series(100 + np.cumsum(rng.normal(0, 1, n)))
        high = close + rng.uniform(0, 1, n)
        sma75_full = sma(close, 75)
        sma200_full = sma(close, 200)
        schedule = pd.DataFrame({"ticker": ["T"] * 2,
                                 "date": [close.index[100], close.index[280]]})
        for t in (250, 280, 300, n - 1):
            full = evaluate_gates(close, high, sma75_full, sma200_full, t, True, 15,
                                  earnings_schedule=schedule, ticker="T")
            c2, h2 = close.iloc[:t + 1], high.iloc[:t + 1]
            cut = evaluate_gates(c2, h2, sma(c2, 75), sma(c2, 200), t, True, 15,
                                 earnings_schedule=schedule, ticker="T")
            self.assertEqual(cut, full, msg=f"t={t}")


class Gate0EarningsScheduleEdgeCasesTest(unittest.TestCase):
    def test_ticker_not_in_schedule_is_unknown(self):
        close, high, sma75, sma200 = _base_arrays()
        close_s = _series(close)
        schedule = pd.DataFrame({"ticker": ["OTHER.T"],
                                 "date": [close_s.index[T] + pd.Timedelta(days=1)]})
        r = evaluate_gates(close_s, _series(high), _series(sma75), _series(sma200),
                           T, True, 15, earnings_schedule=schedule, ticker="1234.T")
        self.assertTrue(r["g0"])
        self.assertTrue(r["g0_earnings_unknown"])

    def test_earnings_exactly_at_label_n_boundary_fails(self):
        close, high, sma75, sma200 = _base_arrays()
        close_s = _series(close)
        asof = close_s.index[T]
        # label_n=15 営業日ちょうど先
        earnings_date = np.busday_offset(np.datetime64(asof, "D"), 15, roll="forward")
        schedule = pd.DataFrame({"ticker": ["1234.T"], "date": [pd.Timestamp(earnings_date)]})
        r = evaluate_gates(close_s, _series(high), _series(sma75), _series(sma200),
                           T, True, 15, earnings_schedule=schedule, ticker="1234.T")
        self.assertFalse(r["g0"])  # 15営業日「以内」なので、ちょうど15日は失敗扱い


if __name__ == "__main__":
    unittest.main()
