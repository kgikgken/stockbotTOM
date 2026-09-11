"""パターン検出のバックテスト（docs/BACKTEST.md）。

固定するのは 4 つ。**ホールドアウトを明示フラグ無しに触らない**、**検定が 10 件で
あること**、**Newey-West を日次平均系列に当てること**、**未来のバーを足しても検出が
変わらないこと**。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.validation import pattern_replay as replay_mod
from stockbot.validation import pattern_report as report_mod
from stockbot.validation.replay import HOLDOUT_WINDOW

K = 3


def series_from(points, n=400, start="2024-01-01"):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    close = np.interp(np.arange(n), xs, ys)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": np.full(n, 1e5)}, index=idx)


def double_bottom_series(n=400):
    """L → H → L → 上抜け を繰り返さない単純な形（成立が 1 回だけ起きる）。"""
    return series_from([(0, 120.0), (300, 100.0), (315, 110.0), (330, 100.5),
                        (345, 118.0), (n - 1, 118.0)], n=n)


def listed_frame(tickers):
    return pd.DataFrame({"ticker": list(tickers), "is_equity": [True] * len(tickers)})


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.df = double_bottom_series()
        self.ohlcv = {"1234.T": self.df}
        self.idx = self.df
        self.listed = listed_frame(["1234.T"])

    def _run(self, tmp, start, end, **kw):
        replay_mod.run(dict(self.ohlcv), self.idx, self.listed, Path(tmp),
                       pd.Timestamp(start), pd.Timestamp(end), K,
                       min_adv_jpy=0.0, min_price=0.0,
                       sectors={"1234.T": "機械"}, log=lambda *_a: None, **kw)
        return replay_mod.load_table(Path(tmp))

    def test_detects_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, "2025-03-01", "2025-06-30")
            self.assertGreater(len(out), 0)
            self.assertEqual(list(out.columns), replay_mod.REPLAY_COLS)
            row = out.iloc[0]
            self.assertEqual(row["pattern"], "double_bottom")
            self.assertTrue(np.isfinite(float(row["close_t"])))
            self.assertTrue(np.isfinite(float(row["atr_t"])))

    def test_only_confirmed_rows_are_saved(self):
        """**監視は保存しない**（成立が事象・監視が状態。PATTERN.md §3.1）。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, "2025-03-01", "2025-06-30")
            self.assertTrue(all(out["breakout_pct"] > 0))

    def test_resumes_without_redoing_a_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, "2025-03-01", "2025-04-30")
            before = sorted(p.name for p in Path(tmp).glob("pattern_replay_*"))
            self._run(tmp, "2025-03-01", "2025-06-30")
            after = sorted(p.name for p in Path(tmp).glob("pattern_replay_*"))
            self.assertTrue(set(before) <= set(after))

    def test_excess_return_subtracts_the_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, "2025-03-01", "2025-06-30")
            r = out.dropna(subset=["r_20", "raw_r_20", "bench_r_20"]).iloc[0]
            self.assertAlmostEqual(float(r["r_20"]),
                                   float(r["raw_r_20"]) - float(r["bench_r_20"]),
                                   places=9)


class HoldoutTest(unittest.TestCase):
    """CLAUDE.md の絶対規則: ホールドアウトは明示フラグ無しで生成しない。"""

    def _series(self):
        # ホールドアウト期間（2026-02〜2026-08）をまたぐ系列
        return series_from([(0, 120.0), (300, 100.0), (315, 110.0), (330, 100.5),
                            (345, 118.0), (499, 118.0)], n=500, start="2024-06-03")

    def test_no_days_inside_the_holdout(self):
        df = self._series()
        with tempfile.TemporaryDirectory() as tmp:
            replay_mod.run({"1234.T": df}, df, listed_frame(["1234.T"]), Path(tmp),
                           pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01"), K,
                           min_adv_jpy=0.0, min_price=0.0, log=lambda *_a: None)
            out = replay_mod.load_table(Path(tmp))
            if len(out):
                inside = out[(out["date"] >= HOLDOUT_WINDOW[0])
                             & (out["date"] < HOLDOUT_WINDOW[1])]
                self.assertEqual(len(inside), 0)
            files = [p.name for p in Path(tmp).glob("pattern_replay_2026-0[2-7]*")]
            self.assertEqual(files, [])

    def test_bars_inside_the_holdout_are_physically_cut(self):
        """**日付で T を絞るだけでは足りない。** 窓の端の T が T+20 で覗く。"""
        df = self._series()
        with tempfile.TemporaryDirectory() as tmp:
            replay_mod.run({"1234.T": df}, df, listed_frame(["1234.T"]), Path(tmp),
                           pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-30"), K,
                           min_adv_jpy=0.0, min_price=0.0, log=lambda *_a: None)
            out = replay_mod.load_table(Path(tmp))
            if len(out):
                # ホールドアウト直前の T は T+20 が切られているので打ち切りになる
                late = out[out["date"] >= pd.Timestamp("2026-01-20")]
                if len(late):
                    self.assertTrue(bool(late["censored"].astype(bool).any()))


class LookaheadTest(unittest.TestCase):
    """未来のバーを足しても、その日の検出と判定値は変わらない（CLAUDE.md）。

    交互スイングの表を全期間で 1 回だけ作る最適化が、未来参照になっていないこと。
    """

    def test_future_bars_do_not_change_the_day(self):
        df = double_bottom_series()
        listed = listed_frame(["1234.T"])
        cut = pd.Timestamp("2025-05-01")
        short = df[df.index <= cut + pd.Timedelta(days=1)]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            replay_mod.run({"1234.T": df}, df, listed, Path(a),
                           pd.Timestamp("2025-04-01"), cut, K,
                           min_adv_jpy=0.0, min_price=0.0, log=lambda *_a: None)
            replay_mod.run({"1234.T": short}, short, listed, Path(b),
                           pd.Timestamp("2025-04-01"), cut, K,
                           min_adv_jpy=0.0, min_price=0.0, log=lambda *_a: None)
            full = replay_mod.load_table(Path(a))
            trimmed = replay_mod.load_table(Path(b))
        cols = ["date", "ticker", "pattern", "neckline", "close_t", "breakout_pct",
                "pattern_low", "target", "span"]
        pd.testing.assert_frame_equal(
            full[cols].reset_index(drop=True), trimmed[cols].reset_index(drop=True),
            check_dtype=False)


class ReportTest(unittest.TestCase):
    """検定 10 件（docs/BACKTEST.md §4）。"""

    def _table(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-08-02", periods=n // 2)
        return pd.DataFrame({
            "date": np.repeat(dates, 2),
            "ticker": [f"{1000 + i}.T" for i in range(n)],
            "pattern": rng.choice(report_mod.TESTED_PATTERNS, n),
            "r_20": rng.normal(0.0, 0.05, n),
            "breakout_h": rng.uniform(0.0, 0.5, n),
            "success": rng.random(n) > 0.5,
            "reached_target": rng.random(n) > 0.7,
            "mfe_atr": rng.normal(2.0, 0.5, n),
            "mae_atr": rng.normal(-1.5, 0.5, n),
        })

    def test_exactly_ten_tests(self):
        """**10 件から増やさない**（§5 の禁止事項）。"""
        self.assertEqual(report_mod.N_TESTS, 10)
        self.assertEqual(len(report_mod.TESTED_PATTERNS), 5)
        self.assertEqual(report_mod.N_QUANTILES, 5)

    def test_flag_and_pennant_are_not_tested(self):
        """C3・C4 は日足かつ k=3 では構造的に 0 件（PATTERN.md D-10）。"""
        self.assertNotIn("bull_flag", report_mod.TESTED_PATTERNS)
        self.assertNotIn("bull_pennant", report_mod.TESTED_PATTERNS)

    def test_pattern_table_keeps_every_group_even_at_zero(self):
        empty = self._table().head(0)
        table = report_mod.by_pattern(empty)
        self.assertEqual(len(table), 5)
        self.assertTrue(all(table["n"] == 0))

    def test_quantile_edges_are_reused_across_windows(self):
        """**境界は探索窓で作り、確認窓に当てる**（D-3）。"""
        search = self._table(seed=1)
        confirm = self._table(seed=2)
        edges = report_mod.quantile_edges(search)
        self.assertEqual(len(edges), 6)
        a = report_mod.by_breakout_quantile(search, edges)
        b = report_mod.by_breakout_quantile(confirm, edges)
        self.assertEqual(list(a["group"]), list(b["group"]))   # 群の定義が同じ
        self.assertEqual(int(a["n"].sum()), len(search))
        self.assertEqual(int(b["n"].sum()), len(confirm))

    def test_daily_mean_series_skips_empty_days(self):
        """**0 件の日は系列に入れない。0 で埋めない**（D-4）。"""
        df = self._table()
        one = df[df["pattern"] == report_mod.TESTED_PATTERNS[0]]
        daily = report_mod.daily_mean_series(one)
        self.assertEqual(len(daily), one["date"].nunique())
        self.assertLess(len(daily), df["date"].nunique())

    def test_newey_west_uses_the_daily_series_not_the_events(self):
        """イベント単位だと重なりで t 値が過大になる。日次平均に当てる。"""
        df = self._table(n=400, seed=3)
        row = report_mod.summarize_group("x", df)
        self.assertEqual(row["n"], 400)
        self.assertEqual(row["n_days"], df["date"].nunique())
        self.assertNotEqual(row["n"], row["n_days"])

    def test_mean_is_the_event_mean(self):
        df = self._table()
        row = report_mod.summarize_group("x", df)
        self.assertAlmostEqual(row["mean_r20"], float(df["r_20"].mean()), places=12)

    def test_diagnostics_are_present_but_separate(self):
        row = report_mod.summarize_group("x", self._table())
        for col in ("win_rate", "success_rate", "target_rate",
                    "mfe_atr_median", "mae_atr_median"):
            self.assertIn(col, row)

    def test_coverage_reports_what_was_actually_evaluated(self):
        df = self._table()
        cov = report_mod.coverage(df)
        self.assertEqual(cov["n_rows"], len(df))
        self.assertEqual(cov["first"], df["date"].min())

    def test_format_table_has_counts(self):
        lines = report_mod.format_table(report_mod.by_pattern(self._table()))
        self.assertIn("件数", lines[0])
        self.assertEqual(len(lines), 6)


if __name__ == "__main__":
    unittest.main()


class UniverseGateTest(unittest.TestCase):
    """バックテストの母集団を運用と揃える（docs/BACKTEST.md §2.1・§8 D-5）。

    **数字をここに書かない。** 運用の `Settings` から受け取る設計になっていること
    （ずれたら再実行が要るので、ずれない形にした）。
    """

    def _prepared(self, adv_per_day, close=1000.0, n=400):
        df = series_from([(0, close), (n - 1, close)], n=n)
        df = df.assign(Volume=np.full(n, adv_per_day / close))
        return replay_mod._prepared({"1234.T": df}, ["1234.T"], K), df

    def test_turnover_gate_drops_a_thin_name(self):
        prep, df = self._prepared(adv_per_day=1e8)     # 1億円
        date_t = df.index[300]
        kept = replay_mod.universe_at(prep, {"1234.T"}, date_t,
                                      min_adv_jpy=2e8, min_price=200.0)
        self.assertEqual(kept, [])
        loose = replay_mod.universe_at(prep, {"1234.T"}, date_t,
                                       min_adv_jpy=5e7, min_price=200.0)
        self.assertEqual(loose, ["1234.T"])

    def test_price_gate_drops_a_low_priced_name(self):
        prep, df = self._prepared(adv_per_day=1e9, close=150.0)
        date_t = df.index[300]
        self.assertEqual(replay_mod.universe_at(prep, {"1234.T"}, date_t,
                                                min_adv_jpy=2e8, min_price=200.0), [])
        self.assertEqual(replay_mod.universe_at(prep, {"1234.T"}, date_t,
                                                min_adv_jpy=2e8, min_price=100.0),
                         ["1234.T"])

    def test_history_gate(self):
        prep, df = self._prepared(adv_per_day=1e9)
        early = df.index[100]                           # 履歴 101 本
        self.assertEqual(replay_mod.universe_at(prep, {"1234.T"}, early, 2e8, 200.0,
                                                min_history_bars=250), [])

    def test_gates_use_only_bars_up_to_t(self):
        """T より後を壊しても、その日のユニバースは変わらない（未来参照の禁止）。"""
        prep, df = self._prepared(adv_per_day=1e9)
        date_t = df.index[300]
        before = replay_mod.universe_at(prep, {"1234.T"}, date_t, 2e8, 200.0)
        broken = df.copy()
        broken.iloc[301:] = broken.iloc[301:] * 0.001
        prep2 = replay_mod._prepared({"1234.T": broken}, ["1234.T"], K)
        self.assertEqual(replay_mod.universe_at(prep2, {"1234.T"}, date_t, 2e8, 200.0),
                         before)


class AlreadyAtTargetTest(unittest.TestCase):
    """**b >= 1 は「最初から目標を超えている」**（docs/BACKTEST.md §3.1）。

    `reached_target` はその行では「届いた」ではなく「最初から届いていた」を数える。
    診断列なので判定には影響しないが、読み違えると危ない。
    """

    def test_b_at_least_one_means_close_is_at_or_above_target(self):
        from stockbot.features.pattern import measured_move

        neck, low = 100.0, 90.0
        height = neck - low
        for b, expected in [(0.5, False), (1.0, True), (1.5, True)]:
            close = neck + b * height
            m = measured_move(neck, close, [low])
            self.assertEqual(close >= m["target"], expected, f"b={b}")

    def test_column_is_set_from_the_detection(self):
        df = double_bottom_series()
        with tempfile.TemporaryDirectory() as tmp:
            replay_mod.run({"1234.T": df}, df, listed_frame(["1234.T"]), Path(tmp),
                           pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30"), K,
                           min_adv_jpy=0.0, min_price=0.0, log=lambda *_a: None)
            out = replay_mod.load_table(Path(tmp))
        self.assertIn("already_at_target", out.columns)
        flag = out["already_at_target"].astype(bool)
        # フラグが立っている行は必ず 終値 >= 目標
        for _i, r in out[flag].iterrows():
            self.assertGreaterEqual(float(r["close_t"]), float(r["target"]))
        for _i, r in out[~flag].iterrows():
            if pd.notna(r["target"]):
                self.assertLess(float(r["close_t"]), float(r["target"]))

    def test_report_shows_the_rate(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2021-08-02", periods=4),
            "r_20": [0.01, -0.01, 0.02, 0.0],
            "already_at_target": [True, True, False, False],
            "reached_target": [True, True, False, True],
        })
        row = report_mod.summarize_group("x", df)
        self.assertAlmostEqual(row["already_rate"], 0.5)
        self.assertIn("既に到達", report_mod.format_table(pd.DataFrame([row]))[0])
