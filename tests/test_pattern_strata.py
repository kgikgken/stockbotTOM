"""既存 4 案の株価帯別・年別の記述統計（docs/BACKTEST.md §13）。**判定ではない。**

固定するのは 6 つ。**検定を 1 件も増やさないこと**、**株価帯の境界が探索窓で作られて
全窓で固定されること**、**現行基準の 3 経路（§10 tgt・§11 current・§12 base）が
一致すること**、**当てはまらない案の行が母数から外れること**、**群が固定で 0 件でも
落ちないこと**、**T+20 より先のバーを足しても結果が変わらないこと**。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features import pattern as pattern_mod
from stockbot.validation import pattern_exit as exit_mod
from stockbot.validation import pattern_stop as stop_mod
from stockbot.validation import pattern_strata as st_mod
from stockbot.validation import pattern_target as tgt_mod


def frame(n=60, price=100.0, start="2024-01-01", highs=None, lows=None):
    h = list(highs) if highs is not None else [price] * n
    lo = list(lows) if lows is not None else [price] * n
    return pd.DataFrame(
        {"Open": [price] * n, "High": h, "Low": lo, "Close": [price] * n,
         "Volume": np.full(n, 1e5)},
        index=pd.bdate_range(start, periods=n))


def row(**kw):
    base = {"pattern": pattern_mod.ASCENDING_TRIANGLE, "t_pos": 30,
            "close_t": 100.0, "neckline": 100.0, "height": 20.0,
            "pattern_low": 80.0, "target": 120.0, "atr_t": 3.0,
            "breakout_h": 0.30, "r_20": 0.01,
            "l1_pos": 0.0, "l1": 80.0, "l2_pos": 10.0, "l2": 85.0,
            "l3_pos": 20.0, "l3": 90.0, "already_at_target": False}
    base.update(kw)
    return base


class TestCountTest(unittest.TestCase):
    def test_no_new_tests(self):
        """**検定は 15 件のまま。ここで 1 件も増やさない**（§13.1）。"""
        self.assertEqual(st_mod.N_TESTS_HERE, 0)
        self.assertEqual(st_mod.N_TESTS_TOTAL, 15)
        self.assertEqual(st_mod.N_TESTS_TOTAL, tgt_mod.N_TESTS_TOTAL)

    def test_three_price_bands(self):
        """**3 分位。振らない**（§13.2）。"""
        self.assertEqual(st_mod.N_PRICE_BANDS, 3)

    def test_variants_are_current_plus_five(self):
        self.assertEqual(st_mod.VARIANTS,
                         ("cur", "atr2", "s1", "s2", "s3", "ratio"))


class BandEdgeTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"close_t": [100.0, 200.0, 300.0, 400.0,
                                            500.0, 600.0]})

    def test_edges_are_tertiles_with_open_ends(self):
        e = st_mod.price_band_edges(self.df)
        self.assertEqual(len(e), 4)
        self.assertEqual((e[0], e[-1]), (-np.inf, np.inf))
        self.assertAlmostEqual(e[1], 266.6666666666667)
        self.assertAlmostEqual(e[2], 433.3333333333333)

    def test_edges_need_close_t(self):
        self.assertIsNone(st_mod.price_band_edges(pd.DataFrame({"x": [1.0]})))
        self.assertIsNone(st_mod.price_band_edges(pd.DataFrame({"close_t": [1.0]})))

    def test_save_and_load_roundtrip(self):
        e = st_mod.price_band_edges(self.df)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bands.json"
            st_mod.save_frozen_bands(e, {"window": "search"}, p)
            back = st_mod.load_frozen_bands(p)
            np.testing.assert_allclose(back, e)
            body = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(body["edges"][0], None)
            self.assertEqual(body["edges"][-1], None)
            self.assertEqual(body["value"], "close_t")

    def test_load_returns_none_when_missing_or_broken(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(st_mod.load_frozen_bands(Path(d) / "nope.json"))
            bad = Path(d) / "bad.json"
            bad.write_text('{"edges": [null, 1.0, null]}', encoding="utf-8")
            self.assertIsNone(st_mod.load_frozen_bands(bad))

    def test_labels_carry_the_boundaries(self):
        labels = st_mod.band_labels(st_mod.price_band_edges(self.df))
        self.assertEqual(len(labels), 3)
        self.assertIn("267", labels[0])
        self.assertIn("433", labels[2])

    def test_top_band_includes_its_upper_end(self):
        """**右端の帯だけ上端を含める**（`by_breakout_quantile` と同じ）。"""
        edges = np.asarray([-np.inf, 200.0, 400.0, np.inf])
        labels = st_mod.band_labels(edges)
        got = st_mod.assign_bucket(np.asarray([100.0, 200.0, 400.0, 1e9]),
                                   edges, labels)
        self.assertEqual(list(got), [labels[0], labels[1], labels[2], labels[2]])

    def test_every_row_lands_in_a_band(self):
        edges = np.asarray([-np.inf, 200.0, 400.0, np.inf])
        got = st_mod.assign_bucket(np.asarray([1.0, 250.0, 9999.0]), edges,
                                   st_mod.band_labels(edges))
        self.assertTrue(all(g != "" for g in got))


class StrataOneTest(unittest.TestCase):
    def setUp(self):
        self.df = frame(n=60, highs=[100.0] * 60, lows=[100.0] * 60)

    def test_current_matches_all_three_sources(self):
        """**現行基準は §10 tgt・§11 current・§12 base と一致する**（§13.5）。"""
        r = row()
        got = st_mod.strata_one(self.df, 30, r)
        e = exit_mod.exit_one(self.df, 30, r)
        s = stop_mod.stop_one(self.df, 30, r)
        t = tgt_mod.target_one(self.df, 30, r)
        self.assertEqual(got["cur_outcome"], e["tgt_outcome"])
        self.assertEqual(got["cur_outcome"], s["current_outcome"])
        self.assertEqual(got["cur_outcome"], t["base_outcome"])
        self.assertAlmostEqual(got["cur_pnl_pct"], e["tgt_pnl_pct"])
        self.assertTrue(got["cur_sources_agree"])

    def test_each_variant_matches_its_own_module(self):
        r = row()
        got = st_mod.strata_one(self.df, 30, r)
        e = exit_mod.exit_one(self.df, 30, r)
        s = stop_mod.stop_one(self.df, 30, r)
        t = tgt_mod.target_one(self.df, 30, r)
        self.assertAlmostEqual(got["atr2_pnl_pct"], e["atr2_pnl_pct"])
        for v in ("s1", "s2", "s3"):
            self.assertAlmostEqual(got[f"{v}_pnl_pct"], s[f"{v}_pnl_pct"], msg=v)
        self.assertAlmostEqual(got["ratio_pnl_pct"], t["ratio_pnl_pct"])

    def test_s3_is_blank_for_a_reversal(self):
        """**S3 は C1・C2 のみ。** 反転系は当てはまらないので空（§11.3）。"""
        got = st_mod.strata_one(self.df, 30, row(pattern=pattern_mod.DOUBLE_BOTTOM))
        self.assertEqual(got["s3_outcome"], "")
        self.assertTrue(pd.isna(got["s3_pnl_pct"]))

    def test_ratio_is_blank_for_a_pennant(self):
        """**ペナントは係数補正の対象外**（§12.2・PATTERN.md Q-5）。"""
        got = st_mod.strata_one(self.df, 30, row(pattern=pattern_mod.BULL_PENNANT))
        self.assertEqual(got["ratio_outcome"], "")
        self.assertNotEqual(got["cur_outcome"], "")

    def test_no_future_reference_beyond_the_horizon(self):
        """**T+20 より先のバーを足しても結果が変わらない**（未来参照の禁止）。"""
        long_df = frame(n=80, highs=[100.0] * 80, lows=[100.0] * 80)
        short_df = long_df.iloc[:51].copy()
        a = st_mod.strata_one(long_df, 30, row(), horizon=20)
        b = st_mod.strata_one(short_df, 30, row(), horizon=20)
        for k in a:
            self.assertEqual(str(a[k]), str(b[k]), k)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.df = frame(n=60)
        self.ohlcv = {"1234.T": self.df, "5678.T": self.df}
        self.edges = np.asarray([-np.inf, 150.0, 400.0, np.inf])
        self.replay = pd.DataFrame([
            {"date": self.df.index[0], "ticker": "1234.T", **row(t_pos=0)},
            {"date": self.df.index[1], "ticker": "5678.T",
             **row(t_pos=1, close_t=500.0, neckline=500.0, height=100.0,
                   pattern_low=400.0, target=600.0, l1=400.0, l2=430.0, l3=460.0)},
        ])

    def test_builds_table_with_band_and_year(self):
        out = st_mod.run(self.replay, self.ohlcv, self.edges, log=lambda *_a: None)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), st_mod.STRATA_COLS)
        self.assertEqual(list(out["year"]), [2024, 2024])
        labels = st_mod.band_labels(self.edges)
        self.assertEqual(list(out["band"]), [labels[0], labels[2]])

    def test_recompute_is_identical(self):
        """再計算一致（DESIGN.md §11）。"""
        a = st_mod.run(self.replay, self.ohlcv, self.edges, log=lambda *_a: None)
        b = st_mod.run(self.replay, self.ohlcv, self.edges, log=lambda *_a: None)
        pd.testing.assert_frame_equal(a, b)

    def test_missing_ohlcv_rows_are_reported_not_dropped_silently(self):
        seen = []
        rep = pd.DataFrame([{"date": self.df.index[0], "ticker": "9999.T",
                             **row(t_pos=0)}])
        out = st_mod.run(rep, self.ohlcv, self.edges, log=seen.append)
        self.assertEqual(len(out), 0)
        self.assertTrue(any("落ちた行" in s for s in seen))

    def test_current_sources_check(self):
        out = st_mod.run(self.replay, self.ohlcv, self.edges, log=lambda *_a: None)
        chk = st_mod.check_current_sources(out)
        self.assertEqual(chk["n"], 2)
        self.assertEqual(chk["n_mismatch"], 0)

    def test_empty(self):
        self.assertEqual(len(st_mod.run(pd.DataFrame(), self.ohlcv, self.edges,
                                        log=lambda *_a: None)), 0)


class SummarizeTest(unittest.TestCase):
    def table(self):
        return pd.DataFrame([
            {"date": pd.Timestamp("2024-01-02"), "s3_outcome": "target",
             "s3_pnl_pct": 10.0, "r_20": 0.02},
            {"date": pd.Timestamp("2024-01-03"), "s3_outcome": "stop",
             "s3_pnl_pct": -4.0, "r_20": -0.01},
            {"date": pd.Timestamp("2024-01-04"), "s3_outcome": "",
             "s3_pnl_pct": np.nan, "r_20": 0.05},
        ])

    def test_rows_the_plan_does_not_apply_to_leave_the_denominator(self):
        """**当てはまらない行は母数から外す**（§11.3・§12.2 と同じ扱い）。"""
        s = st_mod.summarize("x", self.table(), "s3")
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["n_na"], 1)
        self.assertAlmostEqual(s["mean_pnl"], 3.0)
        self.assertAlmostEqual(s["target_rate"], 0.5)
        self.assertAlmostEqual(s["stop_rate"], 0.5)
        self.assertAlmostEqual(s["timeout_rate"], 0.0)
        self.assertAlmostEqual(s["win_rate"], 0.5)

    def test_r20_uses_only_the_rows_in_the_denominator(self):
        """**平均 r20 も母数と同じ行から出す**（対象外の行を混ぜない）。"""
        s = st_mod.summarize("x", self.table(), "s3")
        self.assertAlmostEqual(s["mean_r20"], 0.005)

    def test_empty_group_keeps_its_count(self):
        s = st_mod.summarize("x", pd.DataFrame(), "cur")
        self.assertEqual(s["n"], 0)
        self.assertTrue(np.isnan(s["mean_pnl"]))


class ByStrataTest(unittest.TestCase):
    def setUp(self):
        self.edges = np.asarray([-np.inf, 150.0, 400.0, np.inf])
        self.qedges = np.asarray([-np.inf, 0.1, 0.25, 0.5, 1.0, np.inf])
        self.table = pd.DataFrame([
            {"date": pd.Timestamp("2024-01-02"), "year": 2024,
             "band": st_mod.band_labels(self.edges)[0], "breakout_h": 0.3,
             "r_20": 0.01,
             **{f"{v}_{s}": ("target" if s == "outcome" else 5.0)
                for v in st_mod.VARIANTS for s in ("outcome", "pnl_pct")}},
        ])

    def test_band_axis_keeps_every_band_even_when_empty(self):
        """**群は固定。0 件でも行を落とさない。**"""
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, self.qedges)
        self.assertEqual(list(out.columns), st_mod.SUMMARY_COLS)
        self.assertEqual(sorted(out["bucket"].unique()),
                         sorted(st_mod.band_labels(self.edges)))
        self.assertEqual(set(out["window"]), {"search"})
        self.assertEqual(set(out["axis"]), {st_mod.AXIS_BAND})

    def test_every_plan_appears_in_every_bucket(self):
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, self.qedges)
        for bucket in st_mod.band_labels(self.edges):
            plans = set(out[out["bucket"] == bucket]["plan"])
            self.assertEqual(plans, set(st_mod.PLAN_LABELS.values()), bucket)

    def test_breakout_plan_has_five_quantiles_plus_overall(self):
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, self.qedges)
        part = out[(out["plan"] == st_mod.PLAN_LABELS[st_mod.PLAN_BREAKOUT])
                   & (out["bucket"] == st_mod.band_labels(self.edges)[0])]
        self.assertEqual(len(part), 6)
        self.assertEqual(list(part["group"])[0], "現行基準・全体")

    def test_breakout_plan_without_edges_keeps_only_the_overall_row(self):
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, None)
        part = out[(out["plan"] == st_mod.PLAN_LABELS[st_mod.PLAN_BREAKOUT])
                   & (out["bucket"] == st_mod.band_labels(self.edges)[0])]
        self.assertEqual(len(part), 1)

    def test_year_axis_uses_the_years_present(self):
        out = st_mod.by_strata(self.table, "confirm", st_mod.AXIS_YEAR,
                               self.edges, self.qedges)
        self.assertEqual(sorted(set(out["bucket"])), [2024])
        self.assertEqual(set(out["window"]), {"confirm"})

    def test_format_marks_nw_t_as_reference_only(self):
        """**NW t は参考**であることを見出しに出す（§13.4）。"""
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, self.qedges)
        head = st_mod.format_strata_table(out, st_mod.PLAN_ATR)[0]
        self.assertIn("参考", head)
        self.assertNotIn("平均r20", head)

    def test_format_shows_r20_only_for_the_breakout_plan(self):
        out = st_mod.by_strata(self.table, "search", st_mod.AXIS_BAND,
                               self.edges, self.qedges)
        head = st_mod.format_strata_table(out, st_mod.PLAN_BREAKOUT,
                                          with_r20=True)[0]
        self.assertIn("平均r20", head)


if __name__ == "__main__":
    unittest.main()
