"""撤退基準の検証（docs/BACKTEST.md §11）。

固定するのは 5 つ。**検定が 3 件（合計 14 件）であること**、**S1 が最後の谷である
こと**、**S3 が C1・C2 だけで、検出側と同じ線の引き方であること**、**同日なら撤退**、
**T+20 より先のバーを足しても結果が変わらないこと**。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features import pattern as pattern_mod
from stockbot.validation import pattern_stop as stop_mod


def frame(n=40, price=100.0, start="2024-01-01", highs=None, lows=None):
    h = list(highs) if highs is not None else [price] * n
    lo = list(lows) if lows is not None else [price] * n
    return pd.DataFrame(
        {"Open": [price] * n, "High": h, "Low": lo, "Close": [price] * n,
         "Volume": np.full(n, 1e5)},
        index=pd.bdate_range(start, periods=n))


def row(**kw):
    base = {"pattern": pattern_mod.ASCENDING_TRIANGLE, "t_pos": 30,
            "close_t": 100.0, "neckline": 99.0, "pattern_low": 90.0, "target": 110.0,
            "l1_pos": 0.0, "l1": 90.0, "l2_pos": 10.0, "l2": 92.0,
            "l3_pos": 20.0, "l3": 94.0, "already_at_target": False}
    base.update(kw)
    return base


class TestCountTest(unittest.TestCase):
    def test_three_tests_total_fourteen(self):
        """**検定は既存 11 件 + 3 件 = 14 件**（§11）。増やさない。"""
        self.assertEqual(stop_mod.N_TESTS_HERE, 3)
        self.assertEqual(stop_mod.N_TESTS_TOTAL, 14)

    def test_variants_are_exactly_three_plus_current(self):
        self.assertEqual(stop_mod.STOP_VARIANTS, ("current", "s1", "s2", "s3"))


class LevelTest(unittest.TestCase):
    def test_s1_is_the_last_trough(self):
        """**S1 は最後の谷**（位置が最大のもの）。最安値ではない。"""
        self.assertAlmostEqual(stop_mod.last_trough(row()), 94.0)

    def test_s1_uses_position_not_order(self):
        r = row(l1_pos=20.0, l1=94.0, l2_pos=0.0, l2=90.0, l3_pos=np.nan, l3=np.nan)
        self.assertAlmostEqual(stop_mod.last_trough(r), 94.0)

    def test_s1_is_never_below_the_pattern_low(self):
        """谷の 1 つなので、**必ず最安値以上**になる。"""
        levels = stop_mod.stop_levels(row())
        self.assertGreaterEqual(levels["s1"], levels["current"])

    def test_s1_with_two_troughs(self):
        r = row(l3_pos=np.nan, l3=np.nan)
        self.assertAlmostEqual(stop_mod.last_trough(r), 92.0)

    def test_s1_without_troughs_is_nan(self):
        r = row(l1_pos=np.nan, l1=np.nan, l2_pos=np.nan, l2=np.nan,
                l3_pos=np.nan, l3=np.nan)
        self.assertTrue(np.isnan(stop_mod.last_trough(r)))

    def test_s2_is_the_neckline(self):
        self.assertAlmostEqual(stop_mod.stop_levels(row())["s2"], 99.0)

    def test_s3_triangle_is_the_regression_line_at_t(self):
        """**C1 は回帰直線を T まで延ばす** —— 検出側の `_line_at` と同じ。"""
        r = row()
        pts = [(0.0, 90.0), (10.0, 92.0), (20.0, 94.0)]
        self.assertAlmostEqual(stop_mod.support_line(r),
                               pattern_mod._line_at(pts, 30), places=9)
        self.assertAlmostEqual(stop_mod.support_line(r), 96.0, places=9)

    def test_s3_box_is_the_mean(self):
        """**C2 は平均。** 判定式が「平均からの距離」で水平さを要求しているため。"""
        r = row(pattern=pattern_mod.ASCENDING_BOX)
        self.assertAlmostEqual(stop_mod.support_line(r), 92.0)

    def test_s3_is_nan_for_reversal_patterns(self):
        """**反転系は該当なしとして除外**（§11）。"""
        for p in ("double_bottom", "triple_bottom", "inverse_hs"):
            self.assertTrue(np.isnan(stop_mod.support_line(row(pattern=p))), p)

    def test_s3_nan_for_flag_and_pennant(self):
        for p in ("bull_flag", "bull_pennant"):
            self.assertTrue(np.isnan(stop_mod.support_line(row(pattern=p))), p)

    def test_current_is_the_saved_pattern_low(self):
        self.assertAlmostEqual(stop_mod.stop_levels(row())["current"], 90.0)


class StopOneTest(unittest.TestCase):
    def _run(self, df, r=None, t_pos=30):
        return stop_mod.stop_one(df, t_pos, r or row(), horizon=20)

    def test_gap_is_measured_from_close_t(self):
        """**撤退距離は終値[T] 基準**（カードの down_pct と同じ。§11）。"""
        out = self._run(frame())
        self.assertAlmostEqual(out["current_gap_pct"], -10.0)
        self.assertAlmostEqual(out["s1_gap_pct"], -6.0)
        self.assertAlmostEqual(out["s2_gap_pct"], -1.0)

    def test_tight_stop_is_hit_and_wide_one_is_not(self):
        n = 40
        lows = [100.0] * n
        lows[32] = 95.0          # S1(94) と現行(90) は割らない。S2(99) は割る
        out = self._run(frame(n=n, lows=lows))
        self.assertEqual(out["s2_outcome"], stop_mod.OUTCOME_STOP)
        self.assertEqual(out["current_outcome"], stop_mod.OUTCOME_TIMEOUT)
        self.assertEqual(out["s1_outcome"], stop_mod.OUTCOME_TIMEOUT)
        self.assertAlmostEqual(out["s2_exit_price"], 99.0)

    def test_same_day_is_stop_for_every_variant(self):
        """**同日に撤退と目標なら撤退**（保守的）。4 案とも同じ扱い。"""
        n = 40
        highs, lows = [100.0] * n, [100.0] * n
        highs[32], lows[32] = 120.0, 80.0
        out = self._run(frame(n=n, highs=highs, lows=lows))
        for v in stop_mod.STOP_VARIANTS:
            self.assertEqual(out[f"{v}_outcome"], stop_mod.OUTCOME_STOP, v)

    def test_target_uses_the_existing_strict_definition(self):
        """利確は測定目標のみ。定義は既存の `reached_target` と同じ `High > 目標`。"""
        n = 40
        highs = [100.0] * n
        highs[32] = 110.0           # ちょうど目標。**> ではないので到達しない**
        self.assertEqual(self._run(frame(n=n, highs=highs))["current_outcome"],
                         stop_mod.OUTCOME_TIMEOUT)
        highs[32] = 110.01
        self.assertEqual(self._run(frame(n=n, highs=highs))["current_outcome"],
                         stop_mod.OUTCOME_TARGET)

    def test_s3_row_is_blank_for_reversal(self):
        out = self._run(frame(), row(pattern="inverse_hs"))
        self.assertTrue(np.isnan(out["s3_stop"]))
        self.assertEqual(out["s3_outcome"], "")
        self.assertEqual(out["current_outcome"], stop_mod.OUTCOME_TIMEOUT)

    def test_future_bars_do_not_change_result(self):
        """**未来参照の検査。** T+20 より先を足しても結果が変わらない。"""
        n = 60
        highs, lows = [100.0] * n, [100.0] * n
        for i in range(52, n):
            highs[i], lows[i] = 999.0, 1.0
        long_df = frame(n=n, highs=highs, lows=lows)
        short_df = long_df.iloc[:51].copy()
        a = stop_mod.stop_one(long_df, 30, row(), horizon=20)
        b = stop_mod.stop_one(short_df, 30, row(), horizon=20)
        for k in a:
            self.assertEqual(str(a[k]), str(b[k]), k)

    def test_no_bars_after_t(self):
        out = stop_mod.stop_one(frame(n=31), 30, row(), horizon=20)
        self.assertEqual(out["n_bars"], 0)
        self.assertEqual(out["current_outcome"], "")


class RunTest(unittest.TestCase):
    def setUp(self):
        self.df = frame(n=60)
        self.ohlcv = {"1234.T": self.df}
        self.replay = pd.DataFrame([
            {"date": self.df.index[30], "ticker": "1234.T", **row()},
            {"date": self.df.index[31], "ticker": "1234.T",
             **row(t_pos=31, pattern=pattern_mod.ASCENDING_BOX)},
        ])

    def test_builds_table(self):
        out = stop_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), stop_mod.STOP_COLS)

    def test_recompute_is_identical(self):
        """再計算一致（DESIGN.md §11）。"""
        a = stop_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        b = stop_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        pd.testing.assert_frame_equal(a, b)

    def test_old_generation_without_extremes_is_refused(self):
        """**極値の列が無い再生結果では黙って NaN にしない**（世代のずれを検出する）。"""
        seen = []
        old = self.replay.drop(columns=["l1_pos", "l1", "l2_pos", "l2",
                                        "l3_pos", "l3"])
        out = stop_mod.run(old, self.ohlcv, log=seen.append)
        self.assertEqual(len(out), 0)
        self.assertTrue(any("l1_pos" in s for s in seen))

    def test_missing_ticker_is_reported(self):
        seen = []
        rep = pd.concat([self.replay, pd.DataFrame([
            {"date": self.df.index[30], "ticker": "9999.T", **row()}])],
            ignore_index=True)
        out = stop_mod.run(rep, self.ohlcv, log=seen.append)
        self.assertEqual(len(out), 2)
        self.assertTrue(any("落ちた行" in s for s in seen))

    def test_position_checks(self):
        out = stop_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        chk = stop_mod.check_positions(self.replay, out)
        self.assertEqual(chk["s1_below_low"], 0)
        self.assertEqual(chk["s2_above_close"], 0)


class SummarizeTest(unittest.TestCase):
    def _table(self, rows):
        return pd.DataFrame(rows)

    def test_rows_without_a_stop_are_excluded_from_the_mean(self):
        """**S3 の反転系は母数から外す。** 0 で埋めない。"""
        t = self._table([
            {"date": pd.Timestamp("2024-01-01"), "s3_stop": 95.0,
             "s3_outcome": "target", "s3_pnl_pct": 10.0, "s3_exit_day": 3,
             "s3_gap_pct": -5.0},
            {"date": pd.Timestamp("2024-01-02"), "s3_stop": np.nan,
             "s3_outcome": "", "s3_pnl_pct": np.nan, "s3_exit_day": pd.NA,
             "s3_gap_pct": np.nan},
        ])
        s = stop_mod.summarize_stop("S3", t, "s3")
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["n_no_stop"], 1)
        self.assertAlmostEqual(s["mean_pnl"], 10.0)
        self.assertAlmostEqual(s["target_rate"], 1.0)

    def test_gap_median(self):
        t = self._table([
            {"date": pd.Timestamp(f"2024-01-0{i}"), "s1_stop": 95.0,
             "s1_outcome": "stop", "s1_pnl_pct": -5.0, "s1_exit_day": 2,
             "s1_gap_pct": g} for i, g in enumerate([-3.0, -5.0, -7.0], start=1)])
        self.assertAlmostEqual(stop_mod.summarize_stop("S1", t, "s1")["gap_median"],
                               -5.0)

    def test_daily_mean_not_event_weighted(self):
        """**NW は日次平均系列に当てる**（D-4）。"""
        t = self._table([
            {"date": pd.Timestamp("2024-01-01"), "s1_stop": 95.0,
             "s1_outcome": "stop", "s1_pnl_pct": v, "s1_exit_day": 1,
             "s1_gap_pct": -5.0} for v in (1.0, 3.0)])
        s = stop_mod.summarize_stop("S1", t, "s1")
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["n_days"], 1)

    def test_by_variant_keeps_all_four_rows(self):
        out = stop_mod.by_variant(pd.DataFrame())
        self.assertEqual(len(out), 4)
        self.assertEqual(list(out.columns), stop_mod.STOP_GROUP_COLS)

    def test_format_table(self):
        lines = stop_mod.format_stop_table(stop_mod.by_variant(pd.DataFrame()))
        self.assertEqual(len(lines), 5)
        self.assertIn("S1", lines[2])


if __name__ == "__main__":
    unittest.main()
