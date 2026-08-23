"""価格テンプレート（DESIGN.md §7 / TASKS.md T-302）のテスト。

出来高テンプレートは未実装（docs/TASKS.md 質問ログ T-302、設計責任者の回答待ち）のため
ここでは価格テンプレートのみを対象にする。
"""
import unittest

import numpy as np

from . import _path  # noqa: F401
from stockbot.scoring.template import GRID_N, build_price_grid, price_template_score

L0_LOW = 100.0
H0_HIGH = 200.0
LEG = H0_HIGH - L0_LOW
LEG_BARS = 40  # l0=0, h0=40
D = 40  # t_pos=80
L0, H0, T_POS = 0, LEG_BARS, LEG_BARS + D


def _ideal_close() -> np.ndarray:
    """DESIGN.md §7 の理想経路: 上昇脚は0→1へ対角線的に上昇、押し目は1から0.635へ
    緩やかに下降しつつ最後の40%(≒最後の2列)は水平。"""
    close = np.empty(T_POS + 1)
    for i in range(0, H0):  # 上昇脚 l0..h0-1
        y = i / LEG_BARS
        close[i] = L0_LOW + y * LEG
    close[H0] = H0_HIGH  # h0 自身
    for i in range(H0 + 1, T_POS + 1):  # 押し目
        frac = (i - H0) / D
        y = 1.0 - min(frac, 0.6) / 0.6 * 0.365  # 1.0 -> 0.635 で頭打ち
        close[i] = L0_LOW + y * LEG
    return close


def _broken_close() -> np.ndarray:
    """理想経路をベースに、L0 割れ（y<0）と押し目前半での H0 超え（y>1.0）を混ぜる。"""
    close = _ideal_close().copy()
    # 押し目前半（押し目列の最初の方）で H0 を大きく超える急騰
    spike_i = H0 + 2
    close[spike_i] = H0_HIGH * 1.3
    # 終盤で L0 を大きく割り込む急落
    crash_i = T_POS - 2
    close[crash_i] = L0_LOW * 0.5
    return close


class PriceGridColumnSumTest(unittest.TestCase):
    """必須テスト: 列ごとの P の合計が1。"""

    def test_every_populated_column_sums_to_one(self):
        close = _ideal_close()
        P = build_price_grid(close, L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)
        col_sums = P.sum(axis=0)
        # 81本を10列に配るので全列に足が入るはず
        self.assertTrue((col_sums > 0).all(), msg=col_sums)
        for col in range(GRID_N):
            self.assertAlmostEqual(col_sums[col], 1.0, places=9, msg=f"col={col}")

    def test_undefined_leg_gives_all_nan_grid(self):
        close = _ideal_close()
        P = build_price_grid(close, L0, H0, T_POS, L0_LOW, 0.0, LEG_BARS, D)
        self.assertTrue(np.isnan(P).all())


class IdealVsBrokenPathTest(unittest.TestCase):
    """受け入れ: 合成の理想経路 > 崩れた経路（L0割れ、押し目前半でH0超え）。"""

    def test_ideal_path_scores_higher_than_broken_path(self):
        ideal_score = price_template_score(_ideal_close(), L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)
        broken_score = price_template_score(_broken_close(), L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)
        self.assertFalse(np.isnan(ideal_score))
        self.assertFalse(np.isnan(broken_score))
        self.assertGreater(ideal_score, broken_score)

    def test_score_is_clipped_to_zero_one(self):
        for score in (
            price_template_score(_ideal_close(), L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D),
            price_template_score(_broken_close(), L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D),
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class RecalcConsistencyTest(unittest.TestCase):
    """必須テスト（DESIGN.md §11）: T 時点の値は T より後のデータを追加しても変わらない。"""

    def test_appending_future_bars_does_not_change_score(self):
        close = _ideal_close()
        score_full = price_template_score(close, L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)

        extended = np.concatenate([close, np.array([500.0, 600.0, 1.0, 999.0])])
        score_same_t = price_template_score(extended, L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)

        self.assertEqual(score_full, score_same_t)

    def test_recomputing_with_same_inputs_is_deterministic(self):
        close = _broken_close()
        s1 = price_template_score(close, L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)
        s2 = price_template_score(close, L0, H0, T_POS, L0_LOW, LEG, LEG_BARS, D)
        self.assertEqual(s1, s2)


class EdgeCaseTest(unittest.TestCase):
    def test_zero_or_negative_leg_returns_nan(self):
        close = _ideal_close()
        self.assertTrue(np.isnan(price_template_score(close, L0, H0, T_POS, L0_LOW, 0.0, LEG_BARS, D)))
        self.assertTrue(np.isnan(price_template_score(close, L0, H0, T_POS, L0_LOW, -5.0, LEG_BARS, D)))

    def test_very_short_pullback_still_scores_without_crashing(self):
        # leg_bars=3（構造として許される最小）、d=2（同、最小）
        leg_bars, d = 3, 2
        t_pos = leg_bars + d
        close = np.array([100.0, 133.0, 166.0, 200.0, 190.0, 170.0])
        score = price_template_score(close, 0, leg_bars, t_pos, 100.0, 100.0, leg_bars, d)
        self.assertFalse(np.isnan(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
