"""地合いゲージとブレス（T-207）。合成データで強/中/弱・再計算一致を確認する。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.regime import REGIME_MID, REGIME_STRONG, REGIME_WEAK, compute_breadth, regime_gauge

N = 260


def _idx_series(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, dtype=float, index=idx)


class RegimeGaugeLevelsTest(unittest.TestCase):
    def test_strong_when_all_six_points_hold(self):
        close = _idx_series(100 + np.linspace(0, 60, N))
        r = regime_gauge(close, N - 1, breadth_75=0.7, breadth_200=0.7)
        self.assertEqual(r["level"], REGIME_STRONG)
        self.assertEqual(r["score"], 6)
        self.assertTrue(all(r[k] for k in (
            "close_gt_sma25", "close_gt_sma75", "close_gt_sma200",
            "sma75_rising", "breadth_75_ge_half", "breadth_200_ge_half")))

    def test_weak_when_no_points_hold(self):
        close = _idx_series(160 - np.linspace(0, 60, N))
        r = regime_gauge(close, N - 1, breadth_75=0.2, breadth_200=0.2)
        self.assertEqual(r["level"], REGIME_WEAK)
        self.assertEqual(r["score"], 0)

    def test_mid_when_between_three_and_four_points(self):
        # 長期下落の後に短期急回復: 短期線は上抜けたが長期線(SMA200)はまだ上抜けていない
        close = _idx_series(np.concatenate([np.linspace(150, 90, 220), np.linspace(90, 108, 40)]))
        r = regime_gauge(close, N - 1, breadth_75=0.6, breadth_200=0.3)
        self.assertEqual(r["level"], REGIME_MID)
        self.assertIn(r["score"], (3, 4))
        self.assertFalse(r["close_gt_sma200"])

    def test_score_boundaries(self):
        close = _idx_series(100 + np.linspace(0, 60, N))
        self.assertEqual(regime_gauge(close, N - 1, 1.0, 1.0)["level"], REGIME_STRONG)  # 6
        # breadthを両方外して4点にする
        r4 = regime_gauge(close, N - 1, breadth_75=0.1, breadth_200=0.1)
        self.assertEqual(r4["score"], 4)
        self.assertEqual(r4["level"], REGIME_MID)

    def test_missing_indicators_count_as_not_satisfied(self):
        # 履歴が短くSMA200が定義できない場合、その点は0点扱いで例外にならない
        close = _idx_series(100 + np.linspace(0, 5, 50))
        r = regime_gauge(close, 49, breadth_75=np.nan, breadth_200=np.nan)
        self.assertFalse(r["close_gt_sma200"])
        self.assertFalse(r["breadth_75_ge_half"])
        self.assertFalse(r["breadth_200_ge_half"])
        self.assertIn(r["level"], (REGIME_WEAK, REGIME_MID, REGIME_STRONG))


class ComputeBreadthTest(unittest.TestCase):
    def test_matches_hand_calc(self):
        idx = pd.bdate_range("2024-01-01", periods=250)
        above = pd.DataFrame({"Close": 100 + np.linspace(0, 50, 250)}, index=idx)  # 上昇 -> Close>SMA75/200
        below = pd.DataFrame({"Close": 150 - np.linspace(0, 50, 250)}, index=idx)  # 下降 -> Close<SMA75/200
        short_hist = pd.DataFrame({"Close": np.full(30, 100.0)}, index=idx[-30:])  # SMA定義不可、分母から除外
        ohlcv = {"A": above, "B": above, "C": below, "S": short_hist}
        b75, b200, n200 = compute_breadth(ohlcv, idx[-1])
        self.assertAlmostEqual(b75, 2 / 3, places=9)
        self.assertAlmostEqual(b200, 2 / 3, places=9)
        self.assertEqual(n200, 3)

    def test_empty_universe_returns_nan(self):
        b75, b200, n200 = compute_breadth({}, pd.Timestamp("2024-01-01"))
        self.assertTrue(np.isnan(b75))
        self.assertTrue(np.isnan(b200))
        self.assertEqual(n200, 0)

    def test_recalc_consistency(self):
        idx = pd.bdate_range("2024-01-01", periods=250)
        rng = np.random.default_rng(3)
        ohlcv = {
            f"T{i}": pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(0, 1, 250))}, index=idx)
            for i in range(5)
        }
        for asof_pos in (210, 230, 249):
            asof = idx[asof_pos]
            full = compute_breadth(ohlcv, asof)
            cut = compute_breadth({t: df.iloc[:asof_pos + 1] for t, df in ohlcv.items()}, asof)
            self.assertEqual(full, cut, msg=f"asof_pos={asof_pos}")


class RegimeRecalcConsistencyTest(unittest.TestCase):
    def test_truncated_matches_full(self):
        close = _idx_series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, N)))
        for t in (220, 240, N - 1):
            full = regime_gauge(close, t, 0.55, 0.45)
            cut = regime_gauge(close.iloc[: t + 1], t, 0.55, 0.45)
            self.assertEqual(cut, full, msg=f"t={t}")


if __name__ == "__main__":
    unittest.main()
