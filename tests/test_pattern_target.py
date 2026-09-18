"""測定目標の到達率係数補正（docs/BACKTEST.md §12）。**ホールドアウトで 1 回だけ。**

固定するのは 5 つ。**係数が文献値のまま動かないこと**、**現行が保存済みの target と
一致すること**、**同日なら撤退**、**ホールドアウト以外では走らないこと**、
**T+20 より先のバーを足しても結果が変わらないこと**。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features import pattern as pattern_mod
from stockbot.validation import pattern_target as tgt_mod


def frame(n=40, price=100.0, start="2024-01-01", highs=None, lows=None):
    idx = pd.bdate_range(start, periods=n)
    h = list(highs) if highs is not None else [price] * n
    lo = list(lows) if lows is not None else [price] * n
    return pd.DataFrame({"Open": price, "High": h, "Low": lo, "Close": price,
                         "Volume": 1e6}, index=idx)


def row(**kw):
    base = {"pattern": pattern_mod.ASCENDING_TRIANGLE, "close_t": 100.0,
            "neckline": 100.0, "height": 20.0, "pattern_low": 80.0,
            "target": 120.0, "already_at_target": False}
    base.update(kw)
    return base


class RatioTest(unittest.TestCase):
    def test_coefficients_are_the_literature_values(self):
        """**文献値・変更禁止**（Bulkowski の measure rule）。振って比較しない。"""
        self.assertEqual(tgt_mod.TARGET_RATIO, {
            "double_bottom": 0.73, "triple_bottom": 0.74, "inverse_hs": 0.71,
            "ascending_triangle": 0.70, "ascending_box": 0.78})

    def test_test_count_is_fifteen(self):
        self.assertEqual(tgt_mod.N_TESTS_HERE, 1)
        self.assertEqual(tgt_mod.N_TESTS_TOTAL, 15)

    def test_flag_and_pennant_have_no_coefficient(self):
        """C3・C4 は係数が与えられていない（検出 0 件。PATTERN.md D-10）。"""
        for p in ("bull_flag", "bull_pennant"):
            self.assertTrue(np.isnan(tgt_mod.ratio_for(p)), p)


class TargetsTest(unittest.TestCase):
    def test_base_is_neckline_plus_height(self):
        self.assertAlmostEqual(tgt_mod.targets(row())[tgt_mod.BASE], 120.0)

    def test_ratio_is_neckline_plus_height_times_coefficient(self):
        """上昇三角は 0.70 → 100 + 20×0.70 = 114.0。"""
        self.assertAlmostEqual(tgt_mod.targets(row())[tgt_mod.RATIO], 114.0)

    def test_each_pattern_uses_its_own_coefficient(self):
        for p, k in tgt_mod.TARGET_RATIO.items():
            got = tgt_mod.targets(row(pattern=p))[tgt_mod.RATIO]
            self.assertAlmostEqual(got, 100.0 + 20.0 * k, msg=p)

    def test_ratio_target_is_always_below_base(self):
        """係数はすべて 1.0 未満なので、係数補正は必ず現行より近い。"""
        for p in tgt_mod.TARGET_RATIO:
            t = tgt_mod.targets(row(pattern=p))
            self.assertLess(t[tgt_mod.RATIO], t[tgt_mod.BASE], p)

    def test_no_height_gives_nan(self):
        for bad in (row(height=0.0), row(height=np.nan), row(neckline=np.nan)):
            t = tgt_mod.targets(bad)
            self.assertTrue(np.isnan(t[tgt_mod.BASE]))
            self.assertTrue(np.isnan(t[tgt_mod.RATIO]))

    def test_pattern_without_a_coefficient_has_base_only(self):
        t = tgt_mod.targets(row(pattern="bull_flag"))
        self.assertAlmostEqual(t[tgt_mod.BASE], 120.0)
        self.assertTrue(np.isnan(t[tgt_mod.RATIO]))


class TargetOneTest(unittest.TestCase):
    def _run(self, df, r=None, t_pos=0):
        return tgt_mod.target_one(df, t_pos, r or row(), horizon=20)

    def test_ratio_reached_while_base_is_not(self):
        """**係数補正だけが届く形。** 高値 115 は 114.0 を超え 120.0 には届かない。"""
        n = 40
        highs = [100.0] * n
        highs[5] = 115.0
        out = self._run(frame(n=n, highs=highs))
        self.assertEqual(out["ratio_outcome"], tgt_mod.OUTCOME_TARGET)
        self.assertAlmostEqual(out["ratio_exit_price"], 114.0)
        self.assertEqual(out["base_outcome"], tgt_mod.OUTCOME_TIMEOUT)

    def test_both_reached(self):
        n = 40
        highs = [100.0] * n
        highs[5] = 125.0
        out = self._run(frame(n=n, highs=highs))
        self.assertEqual(out["ratio_outcome"], tgt_mod.OUTCOME_TARGET)
        self.assertEqual(out["base_outcome"], tgt_mod.OUTCOME_TARGET)

    def test_stop_is_the_current_pattern_low_for_both(self):
        """**撤退は現行のまま**（パターン内最安値）。両案で同じ。"""
        n = 40
        lows = [100.0] * n
        lows[5] = 79.0
        out = self._run(frame(n=n, lows=lows))
        for v in tgt_mod.VARIANTS:
            self.assertEqual(out[f"{v}_outcome"], tgt_mod.OUTCOME_STOP, v)
            self.assertAlmostEqual(out[f"{v}_exit_price"], 80.0)

    def test_same_day_is_stop_for_both(self):
        """**同日に撤退と目標なら撤退**（保守的）。"""
        n = 40
        highs, lows = [100.0] * n, [100.0] * n
        highs[5], lows[5] = 130.0, 70.0
        out = self._run(frame(n=n, highs=highs, lows=lows))
        for v in tgt_mod.VARIANTS:
            self.assertEqual(out[f"{v}_outcome"], tgt_mod.OUTCOME_STOP, v)

    def test_target_uses_the_existing_strict_definition(self):
        n = 40
        highs = [100.0] * n
        highs[5] = 114.0            # ちょうど。**> ではないので到達しない**
        self.assertEqual(self._run(frame(n=n, highs=highs))["ratio_outcome"],
                         tgt_mod.OUTCOME_TIMEOUT)
        highs[5] = 114.01
        self.assertEqual(self._run(frame(n=n, highs=highs))["ratio_outcome"],
                         tgt_mod.OUTCOME_TARGET)

    def test_gap_is_measured_from_close_t(self):
        out = self._run(frame())
        self.assertAlmostEqual(out["base_gap_pct"], 20.0)
        self.assertAlmostEqual(out["ratio_gap_pct"], 14.0)

    def test_row_without_a_coefficient_is_blank_on_the_ratio_side(self):
        out = self._run(frame(), row(pattern="bull_flag"))
        self.assertTrue(np.isnan(out["ratio_target"]))
        self.assertEqual(out["ratio_outcome"], "")
        self.assertEqual(out["base_outcome"], tgt_mod.OUTCOME_TIMEOUT)

    def test_future_bars_do_not_change_result(self):
        """**未来参照の検査。** T+20 より先を足しても結果が変わらない。"""
        n = 60
        highs, lows = [100.0] * n, [100.0] * n
        for i in range(22, n):
            highs[i], lows[i] = 999.0, 1.0
        long_df = frame(n=n, highs=highs, lows=lows)
        short_df = long_df.iloc[:21].copy()
        a = tgt_mod.target_one(long_df, 0, row(), horizon=20)
        b = tgt_mod.target_one(short_df, 0, row(), horizon=20)
        for k in a:
            self.assertEqual(str(a[k]), str(b[k]), k)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.df = frame(n=60)
        self.ohlcv = {"1234.T": self.df}
        self.replay = pd.DataFrame([
            {"date": self.df.index[0], "ticker": "1234.T", **row()},
            {"date": self.df.index[1], "ticker": "1234.T",
             **row(pattern=pattern_mod.ASCENDING_BOX)},
        ])

    def test_builds_table(self):
        out = tgt_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), tgt_mod.TARGET_COLS)

    def test_recompute_is_identical(self):
        """再計算一致（DESIGN.md §11）。"""
        a = tgt_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        b = tgt_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        pd.testing.assert_frame_equal(a, b)

    def test_base_matches_the_saved_target(self):
        """**現行の利確値が保存済み target と一致する。** ずれたら引き直しが誤り。"""
        out = tgt_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        chk = tgt_mod.check_base_matches_saved(self.replay, out)
        self.assertEqual(chk["n"], 2)
        self.assertEqual(chk["n_mismatch"], 0)

    def test_mismatch_is_counted(self):
        bad = self.replay.copy()
        bad.loc[0, "target"] = 999.0
        out = tgt_mod.run(bad, self.ohlcv, log=lambda *_a: None)
        self.assertEqual(tgt_mod.check_base_matches_saved(bad, out)["n_mismatch"], 1)

    def test_empty(self):
        self.assertEqual(len(tgt_mod.run(pd.DataFrame(), self.ohlcv,
                                         log=lambda *_a: None)), 0)


class SummarizeTest(unittest.TestCase):
    def test_rows_without_a_target_are_excluded(self):
        t = pd.DataFrame([
            {"date": pd.Timestamp("2026-02-02"), "ratio_target": 114.0,
             "ratio_outcome": "target", "ratio_pnl_pct": 14.0, "ratio_exit_day": 3,
             "ratio_gap_pct": 14.0},
            {"date": pd.Timestamp("2026-02-03"), "ratio_target": np.nan,
             "ratio_outcome": "", "ratio_pnl_pct": np.nan, "ratio_exit_day": pd.NA,
             "ratio_gap_pct": np.nan}])
        s = tgt_mod.summarize_target("x", t, "ratio")
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["n_no_target"], 1)
        self.assertAlmostEqual(s["mean_pnl"], 14.0)

    def test_by_variant_keeps_both_rows(self):
        out = tgt_mod.by_variant(pd.DataFrame())
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["group"]), [tgt_mod.VARIANT_LABELS[v]
                                              for v in tgt_mod.VARIANTS])

    def test_format_table(self):
        lines = tgt_mod.format_target_table(tgt_mod.by_variant(pd.DataFrame()))
        self.assertEqual(len(lines), 3)
        self.assertIn("係数補正", lines[2])


if __name__ == "__main__":
    unittest.main()
