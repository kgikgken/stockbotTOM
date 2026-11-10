"""T+1 終値による早期撤退の記述統計（docs/BACKTEST.md §16）。

固定するのは 6 つ。**検定を 1 件も足さないこと**、**T+1 の中は現行どおりで、
同日に両方なら撤退**、**T+1 の安値が建値を割っても T+1 では降りないこと**
（§16.2 の実装上の読み）、**建値割れの約定が建値ちょうどで損益 0 になること**、
**T+20 より先のバーを足しても結果が変わらないこと**、**両方の案に出口がある行だけを
母数にすること**。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.validation import pattern_early as early_mod
from stockbot.validation.pattern_exit import (OUTCOME_STOP, OUTCOME_TARGET,
                                              OUTCOME_TIMEOUT)


def frame(highs, lows, opens, closes, start="2024-01-01"):
    n = len(highs)
    return pd.DataFrame(
        {"Open": list(opens), "High": list(highs), "Low": list(lows),
         "Close": list(closes), "Volume": np.full(n, 1e5)},
        index=pd.bdate_range(start, periods=n))


def build(bars, n=25):
    """bars は {位置: (Open, High, Low, Close)}。残りは建値の少し上で平らにする。"""
    o, h, lo, c = [], [], [], []
    for i in range(n):
        spec = bars.get(i, (101.0, 105.0, 100.5, 101.0))
        o.append(spec[0]); h.append(spec[1]); lo.append(spec[2]); c.append(spec[3])
    return frame(h, lo, o, c)


def base_row(**kw):
    row = {"close_t": 100.0, "breakout_h": 0.5, "pattern_low": 90.0,
           "target": 120.0, "already_at_target": False}
    row.update(kw)
    return row


ENTRY_BAR = (100.0, 105.0, 95.0, 101.0)   # 建値 100・利確も撤退も起きず終値は建値以上


class TestCount(unittest.TestCase):
    def test_adds_no_test(self):
        self.assertEqual(early_mod.N_TESTS_HERE, 0)

    def test_total_stays_fifteen(self):
        self.assertEqual(early_mod.N_TESTS_TOTAL, 15)


class FirstDayTest(unittest.TestCase):
    """T+1 の中は現行どおり。"""

    def test_stop_on_t1(self):
        df = build({1: (100.0, 105.0, 85.0, 101.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_STOP)
        self.assertEqual(r["early_exit_day"], 1)
        self.assertAlmostEqual(r["early_exit_price"], 90.0)
        self.assertEqual(r["early_outcome"], r["cur_outcome"])

    def test_target_on_t1(self):
        df = build({1: (100.0, 125.0, 99.0, 124.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_TARGET)
        self.assertEqual(r["early_exit_day"], 1)
        self.assertAlmostEqual(r["early_exit_price"], 120.0)

    def test_both_on_t1_is_stop(self):
        """**同日に両方なら撤退**（§10 と同じ流儀）。"""
        df = build({1: (100.0, 125.0, 85.0, 101.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_STOP)

    def test_target_is_strict(self):
        """`High == target` では利確しない（現行の `reached_target` と同じ定義）。"""
        df = build({1: (100.0, 120.0, 99.0, 119.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertNotEqual(r["early_outcome"], OUTCOME_TARGET)


class EarlyExitTest(unittest.TestCase):
    """T+1 の終値が建値を割った行。"""

    def test_exits_at_t2_open(self):
        df = build({1: (100.0, 105.0, 95.0, 99.0), 2: (98.0, 99.0, 97.0, 98.5)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], early_mod.OUTCOME_EARLY)
        self.assertEqual(r["early_exit_day"], 2)
        self.assertAlmostEqual(r["early_exit_price"], 98.0)
        self.assertAlmostEqual(r["early_pnl_pct"], -2.0)
        self.assertTrue(r["early_exit_at_open"])
        self.assertTrue(r["t1_below_entry"])

    def test_equal_to_entry_is_not_below(self):
        """`Close[T+1] == 建値` は「建値以上」側。撤退ラインを引き上げる。"""
        df = build({1: (100.0, 105.0, 95.0, 100.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertFalse(r["t1_below_entry"])
        self.assertNotEqual(r["early_outcome"], early_mod.OUTCOME_EARLY)

    def test_no_next_bar_has_no_outcome(self):
        """T+2 が無ければこの案は当てられない。**0 や終値で埋めない。**"""
        df = build({1: (100.0, 105.0, 95.0, 99.0)}, n=2)
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], "")
        self.assertTrue(pd.isna(r["early_pnl_pct"]))
        self.assertTrue(r["t1_below_entry"])
        # 現行基準のほうは当てられる（時間切れ）。**母数が違うことが分かるようにする**
        self.assertEqual(r["cur_outcome"], OUTCOME_TIMEOUT)


class RaisedStopTest(unittest.TestCase):
    """T+1 の終値が建値以上だった行。"""

    def test_stop_moves_to_entry(self):
        df = build({1: ENTRY_BAR, 3: (101.0, 105.0, 99.0, 100.5)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_STOP)
        self.assertEqual(r["early_exit_day"], 3)
        self.assertAlmostEqual(r["early_exit_price"], 100.0)
        self.assertAlmostEqual(r["early_pnl_pct"], 0.0)

    def test_t1_low_below_entry_does_not_exit_on_t1(self):
        """**引き上げは T+1 の引けを見てから**（§16.2 の実装上の読み）。"""
        df = build({1: (100.0, 105.0, 99.0, 101.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_TIMEOUT)
        self.assertEqual(r["early_exit_day"], 20)

    def test_target_later(self):
        df = build({1: ENTRY_BAR, 5: (101.0, 125.0, 100.5, 124.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_TARGET)
        self.assertEqual(r["early_exit_day"], 5)
        self.assertAlmostEqual(r["early_exit_price"], 120.0)

    def test_timeout_at_last_bar_close(self):
        df = build({1: ENTRY_BAR})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_TIMEOUT)
        self.assertEqual(r["early_exit_day"], 20)
        self.assertAlmostEqual(r["early_exit_price"], float(df["Close"].iloc[20]))

    def test_stop_before_target_wins(self):
        """建値割れが先に来た日のほうを取る。"""
        df = build({1: ENTRY_BAR, 3: (101.0, 105.0, 99.0, 100.5),
                    5: (101.0, 130.0, 100.5, 129.0)})
        r = early_mod.early_one(df, 0, base_row())
        self.assertEqual(r["early_outcome"], OUTCOME_STOP)
        self.assertEqual(r["early_exit_day"], 3)


class NoFutureLeakTest(unittest.TestCase):
    def test_bars_after_horizon_do_not_change_result(self):
        """**T+20 より先のバーを足しても結果が変わらない。**"""
        short = build({1: ENTRY_BAR}, n=21)
        a = early_mod.early_one(short, 0, base_row())
        long = build({1: ENTRY_BAR}, n=40)
        for i in range(21, 40):
            long.iloc[i, long.columns.get_loc("High")] = 300.0
            long.iloc[i, long.columns.get_loc("Low")] = 10.0
        b = early_mod.early_one(long, 0, base_row())
        for k in ("early_outcome", "early_exit_day", "early_exit_price",
                  "early_pnl_pct", "cur_outcome", "cur_pnl_pct"):
            self.assertEqual(a[k], b[k], k)

    def test_recompute_matches(self):
        """再計算一致（DESIGN.md §11）。同じ入力で同じ値になる。"""
        df = build({1: ENTRY_BAR, 3: (101.0, 105.0, 99.0, 100.5)})
        a = early_mod.early_one(df, 0, base_row())
        b = early_mod.early_one(df, 0, base_row())
        self.assertEqual(a, b)


class ScaleCheckTest(unittest.TestCase):
    """再生結果と store の目盛りが合っているか（§16.6）。"""

    def test_matching_scale_passes(self):
        df = build({1: ENTRY_BAR})
        r = early_mod.early_one(df, 0, base_row(close_t=float(df["Close"].iloc[0])))
        self.assertTrue(r["scale_ok"])
        self.assertEqual(r["scale_reason"], "")
        self.assertAlmostEqual(r["store_close_t"], float(df["Close"].iloc[0]))

    def test_close_t_revised_fails(self):
        """分割で store 側だけ調整された行。**T の終値がずれる。**"""
        df = build({1: ENTRY_BAR})
        store_close = float(df["Close"].iloc[0])
        r = early_mod.early_one(df, 0, base_row(close_t=store_close * 3.0))
        self.assertFalse(r["scale_ok"])
        self.assertIn("T の終値", r["scale_reason"])

    def test_tolerance_is_the_store_revision_tolerance(self):
        df = build({1: ENTRY_BAR})
        c = float(df["Close"].iloc[0])
        inside = early_mod.early_one(
            df, 0, base_row(close_t=c * (1 + early_mod.SCALE_TOL * 0.9)))
        outside = early_mod.early_one(
            df, 0, base_row(close_t=c * (1 + early_mod.SCALE_TOL * 1.1)))
        self.assertTrue(inside["scale_ok"])
        self.assertFalse(outside["scale_ok"])

    def test_split_inside_the_window_fails(self):
        """**T ではずれず、評価窓の途中で目盛りが変わる行**も落とす。"""
        bars = {1: ENTRY_BAR}
        for i in range(5, 25):          # T+5 以降を 1/3 にする（分割）
            bars[i] = (33.7, 35.0, 33.5, 33.7)
        df = build(bars)
        r = early_mod.early_one(df, 0, base_row(close_t=float(df["Close"].iloc[0])))
        self.assertFalse(r["scale_ok"])
        self.assertIn("評価窓の中", r["scale_reason"])

    def test_row_is_kept_not_dropped(self):
        """**落とすのは集計から。行は残す**（証拠を残す）。"""
        df = build({1: ENTRY_BAR})
        r = early_mod.early_one(df, 0, base_row(close_t=float(df["Close"].iloc[0]) * 3))
        self.assertFalse(r["scale_ok"])
        self.assertNotEqual(r["early_outcome"], "")


class QuantileTest(unittest.TestCase):
    def setUp(self):
        self.edges = np.asarray([-np.inf, 0.1, 0.25, 0.5, 0.95, np.inf])

    def test_buckets(self):
        got = early_mod.assign_quantile(
            np.asarray([0.0, 0.1, 0.3, 0.6, 2.0]), self.edges)
        self.assertEqual(list(got), ["Q1", "Q2", "Q3", "Q4", "Q5"])

    def test_edge_is_lower_inclusive(self):
        got = early_mod.assign_quantile(np.asarray([0.25, 0.5, 0.95]), self.edges)
        self.assertEqual(list(got), ["Q3", "Q4", "Q5"])

    def test_nan_gets_no_bucket(self):
        got = early_mod.assign_quantile(np.asarray([np.nan]), self.edges)
        self.assertEqual(list(got), [""])


def summary_frame():
    """最後の 1 行は目盛りが合っていない行（集計から外れる）。"""
    return pd.DataFrame([
        # 両方に出口がある 3 行
        {"q": "Q1", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": OUTCOME_TIMEOUT,
         "cur_pnl_pct": 2.0, "early_pnl_pct": 1.0, "cur_exit_day": 20,
         "early_exit_day": 20, "t1_below_entry": False},
        {"q": "Q1", "cur_outcome": OUTCOME_STOP, "early_outcome": early_mod.OUTCOME_EARLY,
         "cur_pnl_pct": -10.0, "early_pnl_pct": -2.0, "cur_exit_day": 8,
         "early_exit_day": 2, "t1_below_entry": True},
        {"q": "Q5", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": OUTCOME_TIMEOUT,
         "cur_pnl_pct": 5.0, "early_pnl_pct": 5.0, "cur_exit_day": 20,
         "early_exit_day": 20, "t1_below_entry": False},
        # 現行だけ当てられた行（母数から外す）
        {"q": "Q1", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": "",
         "cur_pnl_pct": 99.0, "early_pnl_pct": np.nan, "cur_exit_day": 20,
         "early_exit_day": pd.NA, "t1_below_entry": True},
        # 目盛りが合っていない行（**両方に出口はあるが母数から外す**）
        {"q": "Q1", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": OUTCOME_TIMEOUT,
         "cur_pnl_pct": 250.0, "early_pnl_pct": 250.0, "cur_exit_day": 20,
         "early_exit_day": 20, "t1_below_entry": False, "scale_ok": False,
         "scale_reason": "T の終値が再生結果とずれている"},
    ]).fillna({"scale_ok": True, "scale_reason": ""})


class SummarizeTest(unittest.TestCase):
    def test_uses_rows_with_both_outcomes(self):
        r = early_mod.summarize("探索窓", "Q1", summary_frame()[lambda d: d["q"] == "Q1"])
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["n_scale_bad"], 1)
        self.assertEqual(r["n_cur_only"], 1)
        self.assertEqual(r["n_early_only"], 0)
        self.assertAlmostEqual(r["cur_mean_pnl"], -4.0)
        self.assertAlmostEqual(r["early_mean_pnl"], -0.5)
        self.assertAlmostEqual(r["diff"], 3.5)

    def test_breakeven_is_counted_apart_from_win_rate(self):
        """**建値撤退は損益ちょうど 0 で勝率に入らない。** 割合を別に出す。"""
        df = pd.DataFrame([
            {"q": "Q1", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": OUTCOME_STOP,
             "cur_pnl_pct": -3.0, "early_pnl_pct": 0.0, "cur_exit_day": 20,
             "early_exit_day": 5, "t1_below_entry": False},
            {"q": "Q1", "cur_outcome": OUTCOME_TIMEOUT, "early_outcome": OUTCOME_TARGET,
             "cur_pnl_pct": 4.0, "early_pnl_pct": 8.0, "cur_exit_day": 20,
             "early_exit_day": 6, "t1_below_entry": False},
        ])
        r = early_mod.summarize("探索窓", "Q1", df)
        self.assertAlmostEqual(r["early_win_rate"], 0.5)
        self.assertAlmostEqual(r["early_breakeven_rate"], 0.5)

    def test_rates(self):
        r = early_mod.summarize("探索窓", "Q1", summary_frame()[lambda d: d["q"] == "Q1"])
        self.assertAlmostEqual(r["t1_below_rate"], 0.5)
        self.assertAlmostEqual(r["early_rate"], 0.5)
        self.assertAlmostEqual(r["timeout_rate"], 0.5)
        self.assertAlmostEqual(r["early_exit_day_mean"], 11.0)

    def test_empty_group_keeps_row(self):
        r = early_mod.summarize("探索窓", "Q2", summary_frame()[lambda d: d["q"] == "Q2"])
        self.assertEqual(r["n"], 0)
        self.assertTrue(pd.isna(r["diff"]))


class ByQuantileTest(unittest.TestCase):
    def test_excludes_q5_and_adds_total(self):
        out = early_mod.by_quantile(summary_frame(), "探索窓")
        self.assertEqual(list(out["q"]), ["Q1", "Q2", "Q3", "Q4", "Q1〜Q4 計"])
        self.assertNotIn("Q5", list(out["q"]))
        self.assertEqual(int(out[out["q"] == "Q1〜Q4 計"]["n"].iloc[0]), 2)

    def test_excluded_count_is_reported(self):
        got = early_mod.excluded_count(summary_frame())
        self.assertEqual(got["n_excluded"], 1)
        self.assertEqual(got["n_no_quantile"], 0)
        self.assertEqual(got["n_scale_bad"], 1)
        self.assertEqual(sum(got["scale_reasons"].values()), 1)


class PathTest(unittest.TestCase):
    def test_writes_outside_pattern_exit(self):
        """**既存の再生結果ファイルを上書きしない**（§16.1）。"""
        p = early_mod.early_path("data/pattern_early", "search")
        self.assertEqual(p.name, "pattern_early_search.csv.gz")
        self.assertNotIn("pattern_exit", str(p))


if __name__ == "__main__":
    unittest.main()
