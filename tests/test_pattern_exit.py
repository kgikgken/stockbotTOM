"""ATR 基準の出口の検証（docs/BACKTEST.md §10）。

固定するのは 5 つ。**倍率が 2.0 の 1 件だけであること**、**同日に両方なら撤退**、
**「到達」と「超え」の定義を取り違えないこと**、**Newey-West を日次平均系列に
当てること**、**T+20 より先のバーを足しても結果が変わらないこと**。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.validation import pattern_exit as exit_mod
from stockbot.validation.replay import HOLDOUT_WINDOW


def frame(highs, lows, opens=None, closes=None, start="2024-01-01"):
    n = len(highs)
    opens = list(opens) if opens is not None else list(lows)
    closes = list(closes) if closes is not None else list(highs)
    return pd.DataFrame(
        {"Open": opens, "High": list(highs), "Low": list(lows), "Close": closes,
         "Volume": np.full(n, 1e5)},
        index=pd.bdate_range(start, periods=n))


def flat_frame(n=60, price=100.0, start="2024-01-01"):
    return frame([price] * n, [price] * n, [price] * n, [price] * n, start=start)


def base_row(**kw):
    row = {"close_t": 100.0, "atr_t": 5.0, "pattern_low": 90.0, "target": 120.0,
           "already_at_target": False}
    row.update(kw)
    return row


class SimulateExitTest(unittest.TestCase):
    """`simulate_exit` の 4 つの出口。**同日なら撤退**を含む。"""

    def _sim(self, highs, lows, closes, entry=100.0, stop=90.0, target=110.0,
             strict=False):
        h = np.asarray(highs, dtype=float)
        lo = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)
        return exit_mod.simulate_exit(h, lo, c, 0, len(h) - 1, entry, stop, target,
                                      target_strict=strict)

    def test_target_hit(self):
        out = self._sim([100, 105, 112], [99, 99, 99], [100, 100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_TARGET)
        self.assertEqual(out["exit_day"], 3)
        self.assertAlmostEqual(out["exit_price"], 110.0)
        self.assertAlmostEqual(out["pnl_pct"], 10.0)

    def test_stop_hit(self):
        out = self._sim([100, 101, 101], [99, 89, 99], [100, 100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_STOP)
        self.assertEqual(out["exit_day"], 2)
        self.assertAlmostEqual(out["exit_price"], 90.0)
        self.assertAlmostEqual(out["pnl_pct"], -10.0)

    def test_timeout_uses_last_close(self):
        out = self._sim([100, 101, 102], [99, 98, 97], [100, 100, 103.0])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_TIMEOUT)
        self.assertEqual(out["exit_day"], 3)
        self.assertAlmostEqual(out["exit_price"], 103.0)
        self.assertAlmostEqual(out["pnl_pct"], 3.0)

    def test_same_day_is_stop(self):
        """**同日に両方なら撤退**（保守的）。ここが逆だと成績が甘く出る。"""
        out = self._sim([100, 115], [99, 85], [100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_STOP)
        self.assertEqual(out["exit_day"], 2)

    def test_stop_before_target_on_later_day(self):
        out = self._sim([100, 101, 115], [99, 85, 99], [100, 100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_STOP)
        self.assertEqual(out["exit_day"], 2)

    def test_target_before_stop_on_later_day(self):
        out = self._sim([100, 115, 101], [99, 99, 85], [100, 100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_TARGET)
        self.assertEqual(out["exit_day"], 2)

    def test_reach_is_inclusive_but_strict_is_not(self):
        """**「到達」は >=、既存の `reached_target` は >。** 取り違えない。"""
        loose = self._sim([100, 110.0], [99, 99], [100, 100], strict=False)
        strict = self._sim([100, 110.0], [99, 99], [100, 100], strict=True)
        self.assertEqual(loose["outcome"], exit_mod.OUTCOME_TARGET)
        self.assertEqual(strict["outcome"], exit_mod.OUTCOME_TIMEOUT)

    def test_stop_is_strict_below(self):
        """撤退は「割った」＝ `<`。ちょうど同値では切らない（既存の定義と同じ）。"""
        out = self._sim([100, 101], [99, 90.0], [100, 100])
        self.assertEqual(out["outcome"], exit_mod.OUTCOME_TIMEOUT)

    def test_empty_window(self):
        h = np.asarray([100.0])
        out = exit_mod.simulate_exit(h, h, h, 1, 0, 100.0, 90.0, 110.0, False)
        self.assertEqual(out["outcome"], "")
        self.assertTrue(np.isnan(out["pnl_pct"]))

    def test_bad_entry(self):
        out = self._sim([100, 115], [99, 99], [100, 100], entry=0.0)
        self.assertEqual(out["outcome"], "")


class ExitOneTest(unittest.TestCase):
    def test_atr2_price_and_ratio(self):
        df = frame([100] * 30, [100] * 30, [100] * 30, [100] * 30)
        out = exit_mod.exit_one(df, 0, base_row(), horizon=20)
        self.assertAlmostEqual(out["atr_ratio"], 0.05)
        self.assertAlmostEqual(out["atr2_pct"], 10.0)
        self.assertAlmostEqual(out["entry_open"], 100.0)
        self.assertAlmostEqual(out["atr2_price"], 110.0)

    def test_mult_is_two(self):
        """**倍率は 2.0 のみ**（§10 の事前登録）。1.5・2.5 は試さない。"""
        self.assertEqual(exit_mod.ATR_MULT, 2.0)

    def test_atr2_and_target_can_differ(self):
        """ATR×2（110）に届いて測定目標（120）に届かない形。"""
        highs = [100] + [111.0] + [100] * 28
        df = frame(highs, [100] * 30, [100] * 30, [100] * 30)
        out = exit_mod.exit_one(df, 0, base_row(), horizon=20)
        self.assertEqual(out["atr2_outcome"], exit_mod.OUTCOME_TARGET)
        self.assertAlmostEqual(out["atr2_exit_price"], 110.0)
        self.assertEqual(out["tgt_outcome"], exit_mod.OUTCOME_TIMEOUT)

    def test_censored_when_window_short(self):
        df = frame([100] * 6, [100] * 6, [100] * 6, [100] * 6)
        out = exit_mod.exit_one(df, 0, base_row(), horizon=20)
        self.assertEqual(out["n_bars"], 5)
        self.assertTrue(out["censored"])

    def test_no_bars_after_t(self):
        df = frame([100] * 5, [100] * 5, [100] * 5, [100] * 5)
        out = exit_mod.exit_one(df, 4, base_row(), horizon=20)
        self.assertEqual(out["n_bars"], 0)
        self.assertEqual(out["atr2_outcome"], "")

    def test_entry_below_stop_flagged(self):
        """寄り付きが既に撤退ライン割れ。**理想約定の歪みを数えられるようにする。**"""
        df = frame([100] * 30, [80] * 30, [85] * 30, [100] * 30)
        out = exit_mod.exit_one(df, 0, base_row(), horizon=20)
        self.assertTrue(out["entry_below_stop"])

    def test_entry_above_target_flagged(self):
        df = frame([130] * 30, [125] * 30, [125] * 30, [130] * 30)
        out = exit_mod.exit_one(df, 0, base_row(), horizon=20)
        self.assertTrue(out["entry_above_target"])

    def test_missing_atr_keeps_row(self):
        df = frame([100] * 30, [100] * 30, [100] * 30, [100] * 30)
        out = exit_mod.exit_one(df, 0, base_row(atr_t=np.nan), horizon=20)
        self.assertTrue(np.isnan(out["atr2_price"]))
        self.assertEqual(out["atr2_outcome"], "")
        # 測定目標のほうは当てられる
        self.assertEqual(out["tgt_outcome"], exit_mod.OUTCOME_TIMEOUT)

    def test_future_bars_do_not_change_result(self):
        """**未来参照の検査。** T+20 より先を足しても結果が変わらない。"""
        highs = [100] * 25 + [999.0] * 20
        lows = [100] * 25 + [1.0] * 20
        long_df = frame(highs, lows, [100] * 45, [100] * 45)
        short_df = long_df.iloc[:21].copy()
        a = exit_mod.exit_one(long_df, 0, base_row(), horizon=20)
        b = exit_mod.exit_one(short_df, 0, base_row(), horizon=20)
        for key in ("atr2_outcome", "atr2_exit_day", "atr2_exit_price", "atr2_pnl_pct",
                    "tgt_outcome", "tgt_pnl_pct", "n_bars", "censored"):
            self.assertEqual(str(a[key]), str(b[key]), key)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.df = frame([100] * 40, [100] * 40, [100] * 40, [100] * 40)
        self.ohlcv = {"1234.T": self.df}
        self.replay = pd.DataFrame([
            {"date": self.df.index[0], "ticker": "1234.T", "pattern": "ascending_box",
             **base_row()},
            {"date": self.df.index[1], "ticker": "1234.T", "pattern": "double_bottom",
             **base_row()},
        ])

    def test_builds_table(self):
        out = exit_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), exit_mod.EXIT_COLS)

    def test_recompute_is_identical(self):
        """再計算一致（DESIGN.md §11）。2 回走らせて同じ表になる。"""
        a = exit_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        b = exit_mod.run(self.replay, self.ohlcv, log=lambda *_a: None)
        pd.testing.assert_frame_equal(a, b)

    def test_cached_arrays_match_uncached(self):
        """`ohlc_arrays` を渡しても渡さなくても同じ（速度のためだけの引数）。"""
        a = exit_mod.exit_one(self.df, 0, base_row(), horizon=20)
        b = exit_mod.exit_one(self.df, 0, base_row(), horizon=20,
                              arrays=exit_mod.ohlc_arrays(self.df))
        self.assertEqual({k: str(v) for k, v in a.items()},
                         {k: str(v) for k, v in b.items()})

    def test_missing_ticker_is_reported(self):
        seen = []
        rep = pd.concat([self.replay, pd.DataFrame([
            {"date": self.df.index[0], "ticker": "9999.T", "pattern": "inverse_hs",
             **base_row()}])], ignore_index=True)
        out = exit_mod.run(rep, {"1234.T": self.df}, log=seen.append)
        self.assertEqual(len(out), 2)
        self.assertTrue(any("落ちた行" in s for s in seen))

    def test_empty_replay(self):
        out = exit_mod.run(pd.DataFrame(), self.ohlcv, log=lambda *_a: None)
        self.assertEqual(len(out), 0)

    def test_holdout_is_truncated(self):
        """**ホールドアウトは物理的に打ち切る**（CLAUDE.md の絶対規則）。"""
        idx = pd.bdate_range("2026-01-20", periods=40)
        df = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
                           "Volume": 1e5}, index=idx)
        cut = exit_mod.truncate_before_holdout({"1234.T": df})
        self.assertTrue((cut["1234.T"].index < HOLDOUT_WINDOW[0]).all())


class StatsTest(unittest.TestCase):
    def test_ratio_quartiles(self):
        table = pd.DataFrame({"atr_ratio": [0.01, 0.02, 0.03, 0.04, 0.05]})
        s = exit_mod.atr_ratio_stats(table)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["median"], 0.03)
        self.assertAlmostEqual(s["min"], 0.01)
        self.assertAlmostEqual(s["max"], 0.05)

    def test_ratio_stats_empty(self):
        self.assertEqual(exit_mod.atr_ratio_stats(pd.DataFrame())["n"], 0)

    def test_format_shows_both_rows(self):
        table = pd.DataFrame({"atr_ratio": [0.02, 0.03]})
        lines = exit_mod.format_ratio_stats("search", exit_mod.atr_ratio_stats(table))
        self.assertEqual(len(lines), 3)
        self.assertIn("×2.0", lines[2])


class SummarizeTest(unittest.TestCase):
    def _table(self, rows):
        return pd.DataFrame(rows)

    def test_rates_and_mean(self):
        t = self._table([
            {"date": pd.Timestamp("2024-01-01"), "atr2_outcome": "target",
             "atr2_pnl_pct": 10.0, "atr2_exit_day": 3},
            {"date": pd.Timestamp("2024-01-02"), "atr2_outcome": "stop",
             "atr2_pnl_pct": -10.0, "atr2_exit_day": 5},
            {"date": pd.Timestamp("2024-01-03"), "atr2_outcome": "timeout",
             "atr2_pnl_pct": 2.0, "atr2_exit_day": 20},
        ])
        s = exit_mod.summarize_exit("x", t)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["n_days"], 3)
        self.assertAlmostEqual(s["mean_pnl"], 2.0 / 3)
        for key in ("target_rate", "stop_rate", "timeout_rate"):
            self.assertAlmostEqual(s[key], 1 / 3)
        self.assertAlmostEqual(s["exit_day_median"], 5.0)

    def test_daily_mean_not_event_weighted(self):
        """**NW は日次平均系列に当てる**（D-4）。同じ日の行は先に平均する。"""
        same_day = self._table([
            {"date": pd.Timestamp("2024-01-01"), "atr2_outcome": "target",
             "atr2_pnl_pct": v, "atr2_exit_day": 1} for v in (1.0, 3.0)])
        s = exit_mod.summarize_exit("x", same_day)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["n_days"], 1)

    def test_empty(self):
        s = exit_mod.summarize_exit("x", pd.DataFrame())
        self.assertEqual(s["n"], 0)
        self.assertTrue(np.isnan(s["mean_pnl"]))

    def test_format_table(self):
        t = self._table([
            {"date": pd.Timestamp("2024-01-01"), "atr2_outcome": "target",
             "atr2_pnl_pct": 10.0, "atr2_exit_day": 3}])
        rows = pd.DataFrame([exit_mod.summarize_exit("ATR×2.0", t)])
        lines = exit_mod.format_exit_table(rows[exit_mod.EXIT_GROUP_COLS])
        self.assertEqual(len(lines), 2)
        self.assertIn("ATR×2.0", lines[1])


class TestCountTest(unittest.TestCase):
    def test_total_is_eleven(self):
        """**検定は既存 10 件 + 今回 1 件 = 11 件**（§10）。増やさない。"""
        self.assertEqual(exit_mod.N_TESTS_HERE, 1)
        self.assertEqual(exit_mod.N_TESTS_TOTAL, 11)


class NoExitTest(unittest.TestCase):
    def test_undefined_target_is_blank_not_timeout(self):
        """利確の値が無い行を「時間切れ」に数えない（数えると成績が動く）。"""
        h = np.asarray([100.0, 101.0])
        out = exit_mod.simulate_exit(h, h, h, 0, 1, 100.0, 90.0, float("nan"), False)
        self.assertEqual(out["outcome"], "")

    def test_rates_use_n_as_denominator(self):
        t = pd.DataFrame([
            {"date": pd.Timestamp("2024-01-01"), "atr2_outcome": "target",
             "atr2_pnl_pct": 10.0, "atr2_exit_day": 3},
            {"date": pd.Timestamp("2024-01-02"), "atr2_outcome": "",
             "atr2_pnl_pct": np.nan, "atr2_exit_day": pd.NA},
        ])
        s = exit_mod.summarize_exit("x", t)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["n_no_exit"], 1)
        self.assertAlmostEqual(s["target_rate"], 0.5)
        self.assertAlmostEqual(s["mean_pnl"], 10.0)


class PathTest(unittest.TestCase):
    def test_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(exit_mod.exit_path(Path(tmp), "search").name,
                             "pattern_exit_search.csv.gz")


if __name__ == "__main__":
    unittest.main()
