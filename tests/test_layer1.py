"""L1 集計（DESIGN.md §10.2 の 0〜6 / TASKS.md T-403）のテスト。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.validation import layer1, replay
from stockbot.validation.labels import LABEL_FAILURE, LABEL_SUCCESS, LABEL_UNDETERMINED


class NeweyWestTest(unittest.TestCase):
    """手計算と一致することを確認（DESIGN.md §10.1 のNewey-West補正）。"""

    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_lag_zero_matches_plain_population_se(self):
        out = layer1.newey_west_mean_t(self.X, lag=0)
        self.assertAlmostEqual(out["mean"], 3.0, places=9)
        self.assertAlmostEqual(out["se"], np.sqrt(2.0 / 5), places=9)
        self.assertAlmostEqual(out["t"], 3.0 / np.sqrt(2.0 / 5), places=9)
        self.assertEqual(out["n"], 5)

    def test_lag_one_matches_hand_calc(self):
        # gamma0=2.0, gamma1=0.8, w=0.5 -> var=2.0+2*0.5*0.8=2.8, se=sqrt(2.8/5)
        out = layer1.newey_west_mean_t(self.X, lag=1)
        self.assertAlmostEqual(out["se"], np.sqrt(2.8 / 5), places=9)
        self.assertAlmostEqual(out["t"], 3.0 / np.sqrt(2.8 / 5), places=9)

    def test_too_few_points_gives_nan(self):
        out = layer1.newey_west_mean_t(np.array([1.0]), lag=0)
        self.assertTrue(np.isnan(out["t"]))

    def test_nan_values_are_dropped(self):
        out = layer1.newey_west_mean_t(np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0]), lag=0)
        self.assertEqual(out["n"], 5)
        self.assertAlmostEqual(out["mean"], 3.0, places=9)


class BhFdrTest(unittest.TestCase):
    def test_matches_hand_calc(self):
        p = np.array([0.005, 0.01, 0.03, 0.5])
        q = layer1.bh_fdr_qvalues(p)
        np.testing.assert_allclose(q, [0.02, 0.02, 0.04, 0.5])

    def test_nan_pvalues_stay_nan(self):
        p = np.array([0.01, np.nan, 0.5])
        q = layer1.bh_fdr_qvalues(p)
        self.assertTrue(np.isnan(q[1]))
        self.assertFalse(np.isnan(q[0]))
        self.assertFalse(np.isnan(q[2]))

    def test_all_nan_returns_all_nan(self):
        q = layer1.bh_fdr_qvalues([np.nan, np.nan])
        self.assertTrue(np.isnan(q).all())


class ClassifyTTest(unittest.TestCase):
    """DESIGN.md §10.2-5: t>3採用、2〜3保留、2未満棄却（分類ラベルのみ、採否はしない）。"""

    def test_boundaries(self):
        self.assertEqual(layer1.classify_t(3.01), "採用")
        self.assertEqual(layer1.classify_t(3.0), "保留")
        self.assertEqual(layer1.classify_t(2.0), "保留")
        self.assertEqual(layer1.classify_t(1.99), "棄却")
        self.assertEqual(layer1.classify_t(-5.0), "棄却")
        self.assertEqual(layer1.classify_t(np.nan), "判定不能")


class DepthBandTest(unittest.TestCase):
    def test_boundaries(self):
        s = pd.Series([0.0, 0.049, 0.05, 0.099, 0.10, 0.15])
        bands = layer1.depth_band(s).astype(str)
        self.assertEqual(list(bands), ["<5%", "<5%", "5-10%", "5-10%", ">10%", ">10%"])


class WriteTableTest(unittest.TestCase):
    """受け入れ: すべての表がcsvとmarkdownで出る。"""

    def test_writes_both_csv_and_markdown(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with TemporaryDirectory() as tmp:
            paths = layer1.write_table(df, tmp, "sample")
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["md"].exists())
            back = pd.read_csv(paths["csv"])
            self.assertEqual(list(back.columns), ["a", "b"])
            md = paths["md"].read_text()
            self.assertIn("| a | b |", md)
            self.assertIn("| 1 | x |", md)

    def test_empty_dataframe_still_writes_header(self):
        df = pd.DataFrame(columns=["a", "b"])
        with TemporaryDirectory() as tmp:
            paths = layer1.write_table(df, tmp, "empty")
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["md"].exists())


class TestCounterTest(unittest.TestCase):
    def test_counts_records(self):
        c = layer1.TestCounter()
        self.assertEqual(c.count, 0)
        c.record("univariate", "d1_ma_stack")
        c.record("dimension_ic", "D1")
        self.assertEqual(c.count, 2)
        frame = c.to_frame()
        self.assertEqual(list(frame.columns), ["kind", "name"])
        self.assertEqual(len(frame), 2)


def _pool(rows: list[dict]) -> pd.DataFrame:
    base_cols = {"ticker": "1301.T", "date": pd.Timestamp("2024-01-01"), "state": "形成中",
                "regime": "中", "depth_pct": 0.05, "r_10": 0.0, "label": LABEL_UNDETERMINED}
    out = []
    for r in rows:
        row = dict(base_cols)
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)


class SuccessRateTest(unittest.TestCase):
    def test_excludes_undetermined_from_denominator(self):
        s = pd.Series([LABEL_SUCCESS, LABEL_SUCCESS, LABEL_FAILURE, LABEL_UNDETERMINED])
        self.assertAlmostEqual(layer1._success_rate(s), 2 / 3, places=9)

    def test_no_decided_rows_gives_nan(self):
        s = pd.Series([LABEL_UNDETERMINED, LABEL_UNDETERMINED])
        self.assertTrue(np.isnan(layer1._success_rate(s)))


class UnivariateDecileCurveTest(unittest.TestCase):
    def test_binary_feature_grouped_by_value_not_qcut(self):
        pool = _pool([
            {"ticker": "A", "d3_bad_news": True, "r_10": -0.05, "label": LABEL_FAILURE},
            {"ticker": "B", "d3_bad_news": True, "r_10": -0.03, "label": LABEL_FAILURE},
            {"ticker": "C", "d3_bad_news": False, "r_10": 0.02, "label": LABEL_SUCCESS},
            {"ticker": "D", "d3_bad_news": False, "r_10": 0.04, "label": LABEL_SUCCESS},
        ])
        curve = layer1.univariate_decile_curve(pool, "d3_bad_news", h=10)
        self.assertEqual(set(curve["bucket"]), {"True", "False"})
        true_row = curve[curve["bucket"] == "True"].iloc[0]
        self.assertAlmostEqual(true_row["mean_r_10"], -0.04, places=9)
        self.assertEqual(true_row["success_rate"], 0.0)

    def test_missing_feature_returns_empty(self):
        pool = _pool([{"ticker": "A"}])
        curve = layer1.univariate_decile_curve(pool, "does_not_exist", h=10)
        self.assertEqual(len(curve), 0)


class IcSummaryTest(unittest.TestCase):
    def test_strong_rank_correlation_gives_high_ic_and_adopt_classification(self):
        # スコアの順位をほぼそのままr_10に反映（少しノイズを混ぜて日ごとのIC分散を
        # 非ゼロにする。完全に毎日IC=1.0だと分散0でt値が定義できない退化例になるため）
        rng = np.random.default_rng(0)
        rows = []
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for day in range(10):
            noise = rng.normal(0, 0.3, size=len(scores))
            for i, score in enumerate(scores):
                rows.append({"ticker": f"T{i}", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                            "score_v1": score, "r_10": score * 10 + noise[i], "label": LABEL_SUCCESS})
        pool = pd.DataFrame(rows)
        summary = layer1.ic_summary(pool, "score_v1", h=10)
        self.assertGreater(summary["ic_mean"], 0.5)
        self.assertEqual(summary["t_classification"], layer1.classify_t(summary["ic_t"]))
        self.assertGreater(summary["q5_mean_r"], summary["q1_mean_r"])


class BaselineTest(unittest.TestCase):
    def test_baseline_c_equal_weight_matches_mean(self):
        pool = _pool([
            {"r_10": 0.1, "label": LABEL_SUCCESS},
            {"r_10": -0.2, "label": LABEL_FAILURE},
            {"r_10": 0.3, "label": LABEL_SUCCESS},
        ])
        out = layer1.baseline_c_equal_weight(pool, h=10)
        self.assertAlmostEqual(out["mean_r"], (0.1 - 0.2 + 0.3) / 3, places=9)
        self.assertAlmostEqual(out["success_rate"], 2 / 3, places=9)


class NishimuraTradesTest(unittest.TestCase):
    """DESIGN.md §10.2-4(b): エントリー/手仕舞い条件の手計算。"""

    def _series(self, closes: list[float]) -> dict:
        idx = pd.bdate_range("2024-01-01", periods=len(closes))
        close = pd.Series(closes, index=idx)
        # Open は Close と同値にしておき、寄付き価格の検証をしやすくする
        df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                           "Close": close}, index=idx)
        return df

    def test_entry_and_sma5_recovery_exit(self):
        # 上昇トレンド（sma75がcloseより十分下）を作った後に一時的な下落を入れる。
        # dip日はsma5を大きく下回りつつsma75より上（エントリー条件成立）。
        # その後closeがsma5を上回った日の翌日寄付きで手仕舞いされるはず
        up = list(np.linspace(70.0, 100.0, 80))
        dip = [96.0, 90.0]  # 90 がエントリー条件を満たす日
        recover = [92.0, 101.0, 102.0]  # 101 で sma5 を上回る -> 翌日(102)の寄付きで手仕舞い
        closes = up + dip + recover + [102.0] * 15
        df = self._series(closes)
        trades = layer1.nishimura_trades({"1301.T": df}, ["1301.T"],
                                         df.index[0], df.index[-1], max_hold=15)
        self.assertEqual(len(trades), 1)
        t = trades.iloc[0]
        self.assertEqual(t["exit_reason"], "sma5_recover")
        self.assertAlmostEqual(t["entry_price"], 92.0, places=9)
        self.assertAlmostEqual(t["exit_price"], 102.0, places=9)
        self.assertAlmostEqual(t["return"], 102.0 / 92.0 - 1, places=9)

    def test_no_entry_when_below_sma75(self):
        # close <= 0.95*sma5 だが close <= sma75（下降トレンド）なので発火しないはず
        n = 100
        closes = list(np.linspace(150.0, 50.0, n))  # 一貫した下降
        df = self._series(closes)
        trades = layer1.nishimura_trades({"1301.T": df}, ["1301.T"], df.index[0], df.index[-1])
        self.assertEqual(len(trades), 0)

    def test_summary_computes_win_rate_and_profit_factor(self):
        trades = pd.DataFrame({"return": [0.1, 0.2, -0.1], "hold_days": [5, 6, 7]})
        summary = layer1.nishimura_summary(trades)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3, places=9)
        self.assertAlmostEqual(summary["profit_factor"], 0.3 / 0.1, places=9)
        self.assertEqual(summary["n_trades"], 3)

    def test_summary_empty_trades(self):
        summary = layer1.nishimura_summary(pd.DataFrame(columns=["return", "hold_days"]))
        self.assertEqual(summary["n_trades"], 0)
        self.assertTrue(np.isnan(summary["win_rate"]))


class SanityTablesTest(unittest.TestCase):
    def test_detects_high_correlation_pair_without_dropping_anything(self):
        n = 30
        x = np.linspace(0, 1, n)
        pool = pd.DataFrame({
            "d1_ma_stack": x,
            "d1_slope200": x * 2 + 1,  # d1_ma_stackと完全な順位相関
            "d1_slope75": np.random.default_rng(0).normal(size=n),
        })
        out = layer1.sanity_tables(pool, feature_ids=["d1_ma_stack", "d1_slope200", "d1_slope75"])
        pairs = out["high_correlation_pairs"]
        self.assertTrue(((pairs["feature_a"] == "d1_ma_stack") & (pairs["feature_b"] == "d1_slope200")).any())
        # 「落とす」ことはしない: 元の特徴量はどの表にも残っている
        self.assertIn("d1_ma_stack", out["missing_rate"]["feature_id"].tolist())
        self.assertIn("d1_slope200", out["missing_rate"]["feature_id"].tolist())

    def test_missing_rate_is_computed(self):
        pool = pd.DataFrame({"d1_ma_stack": [1.0, np.nan, 3.0, np.nan]})
        out = layer1.sanity_tables(pool, feature_ids=["d1_ma_stack"])
        row = out["missing_rate"].iloc[0]
        self.assertEqual(row["n_missing"], 2)
        self.assertAlmostEqual(row["missing_rate"], 0.5, places=9)

    def test_empty_pool_still_has_named_columns_not_a_headerless_csv(self):
        # 回帰テスト: 行が0件でも列名だけは必ず持つこと。そうでないと to_csv が
        # 0バイトのファイルを書いてしまい、後段の read_csv が EmptyDataError で落ちる
        # （report.build_l1_report が実データで踏んだ不具合）
        out = layer1.sanity_tables(pd.DataFrame(), feature_ids=[])
        self.assertEqual(list(out["distribution"].columns), layer1.DISTRIBUTION_COLS)
        self.assertEqual(list(out["missing_rate"].columns), layer1.MISSING_RATE_COLS)
        with TemporaryDirectory() as tmp:
            paths = layer1.write_table(out["distribution"], tmp, "empty_distribution")
            # 0バイトにならず、pandasがヘッダだけの表として読み戻せること
            back = pd.read_csv(paths["csv"])
            self.assertEqual(list(back.columns), layer1.DISTRIBUTION_COLS)
            self.assertEqual(len(back), 0)


class SurvivorshipNoteTest(unittest.TestCase):
    def test_computes_ratio(self):
        delistings = pd.DataFrame({"ticker": ["9999.T", "8888.T"]})
        listed = pd.DataFrame({"ticker": ["1301.T", "1302.T", "1303.T"], "is_equity": [True, True, True]})
        note = layer1.survivorship_note(delistings, listed)
        self.assertEqual(note["n_delisted"], 2)
        self.assertEqual(note["n_current_universe"], 3)
        self.assertAlmostEqual(note["delisted_ratio"], 2 / 5, places=9)

    def test_missing_inputs_gives_nan(self):
        note = layer1.survivorship_note(None, None)
        self.assertTrue(np.isnan(note["n_delisted"]))


def _extend(df: pd.DataFrame, n_extra: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    last_close = float(df["Close"].iloc[-1])
    idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=n_extra)
    ret = rng.normal(0.0002, 0.015, n_extra)
    close = last_close * np.exp(np.cumsum(ret))
    o = close * (1 + rng.normal(0, 0.004, n_extra))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.005, n_extra)))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.005, n_extra)))
    v = rng.lognormal(mean=np.log(5e5), sigma=0.3, size=n_extra)
    ext = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": close, "Volume": v,
                        "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    ext.index.name = "Date"
    return pd.concat([df, ext])


class RunLayer1IntegrationTest(unittest.TestCase):
    """受け入れ: すべての表がcsvとmarkdownで出る。検定数が自動で数えられる（統合確認）。"""

    def test_all_tables_written_in_both_formats_and_test_count_recorded(self):
        n_bars = 320
        replay_end = pd.Timestamp("2021-08-16")
        tickers = [f"{1301 + i * 37:04d}.T" for i in range(20)]
        ohlcv_upto = make_synthetic(tickers, n_bars=n_bars, seed=0, end=replay_end)
        idx_ohlcv = make_synthetic_index(n_bars, seed=0, end=replay_end)
        ohlcv = {t: _extend(ohlcv_upto[t], 30, int(t[:4])) for t in tickers}
        listed = pd.DataFrame({"ticker": tickers, "is_equity": True,
                               "market": ["プライム"] * 10 + ["スタンダード"] * 10})

        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, pd.Timestamp("2021-08-10"), replay_end,
                              k=3, label_n=15, pool_days=5, min_history_bars=250, log=lambda *a: None)
            pool = replay.load_replay_table(tmp)
            self.assertGreater(len(pool), 0)

            out_dir = Path(tmp) / "l1"
            counter = layer1.run_layer1(
                pool, out_dir, ohlcv=ohlcv, listed=listed, h=10, n_draws=3, seed=0,
                nishimura_start=pd.Timestamp("2021-08-10"), nishimura_end=replay_end,
                log=lambda *a: None)

            self.assertGreater(counter.count, 0)
            csv_files = sorted(out_dir.glob("*.csv"))
            md_files = sorted(out_dir.glob("*.md"))
            self.assertGreater(len(csv_files), 0)
            # 全ての表が csv と markdown の両方で出ている（拡張子違いのペアがすべて揃う）
            csv_stems = {f.stem for f in csv_files}
            md_stems = {f.stem for f in md_files}
            self.assertEqual(csv_stems, md_stems)

            # 検定数ログ自体もひとつの表として出ている
            self.assertIn("5_test_count_log", csv_stems)
            log_df = pd.read_csv(out_dir / "5_test_count_log.csv")
            self.assertEqual(len(log_df), counter.count)


if __name__ == "__main__":
    unittest.main()
