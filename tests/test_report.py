"""L1 レポート（DESIGN.md §10.2/§10.3 / TASKS.md T-405）のテスト。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.validation import calibration, layer1, replay, report


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


class BuildL1ReportUnitTest(unittest.TestCase):
    """layer1 の出力を手作りして、束ね方だけを検証する（軽量・高速）。"""

    def _write_minimal_layer1_output(self, out_dir: Path) -> None:
        layer1.write_table(pd.DataFrame({"feature_id": ["d1_ma_stack"], "n": [10],
                                         "n_missing": [1], "missing_rate": [0.1]}),
                           out_dir, "0_sanity_missing_rate")
        layer1.write_table(pd.DataFrame(columns=["feature_a", "feature_b", "rho"]),
                           out_dir, "0_sanity_high_correlation_pairs")
        layer1.write_table(pd.DataFrame({"feature_id": ["d1_ma_stack"], "bucket": ["1"],
                                         "n": [5], "mean_r_10": [0.01], "success_rate": [0.5]}),
                           out_dir, "1_univariate_deciles")
        layer1.write_table(pd.DataFrame({"score": ["dim_D1_score"], "ic_mean": [0.1], "ic_t": [2.5]}),
                           out_dir, "2_dimension_ic")
        layer1.write_table(pd.DataFrame({"score": ["score_v1"], "ic_mean": [0.15], "ic_t": [4.0]}),
                           out_dir, "3_composite_ic")
        for v in layer1.VARIANTS:
            layer1.write_table(pd.DataFrame({"feature_id": [f"score_{v}"], "bucket": ["1"],
                                             "n": [5], "mean_r_10": [0.02], "success_rate": [0.6]}),
                               out_dir, f"3_composite_{v}_deciles")
            for cut in layer1.CUT_DEFINITIONS:
                layer1.write_table(pd.DataFrame({"variant": [v], "cut": [cut], "group": ["形成中"],
                                                 "n_rows": [5], "ic_mean": [0.1]}),
                                   out_dir, f"3_composite_by_{v}_{cut}")
        layer1.write_table(pd.DataFrame({"baseline": ["c_equal_weight"], "mean_r": [0.01]}),
                           out_dir, "4_baseline_comparison")
        layer1.write_table(pd.DataFrame({"kind": ["univariate"], "name": ["d1_ma_stack"]}),
                           out_dir, "5_test_count_log")
        layer1.write_table(pd.DataFrame({"n_delisted": [3], "n_current_universe": [100]}),
                           out_dir, "6_survivorship_note")

    def test_header_states_test_count_and_no_holdout(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path, window_label="2021-08〜2026-02")
            text = report_path.read_text()
            self.assertIn("検定数: 1", text)
            self.assertIn("ホールドアウト使用: なし", text)
            self.assertIn("2021-08〜2026-02", text)

    def test_explicit_test_counter_overrides_log_file_count(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            counter = layer1.TestCounter()
            counter.record("univariate", "a")
            counter.record("univariate", "b")
            counter.record("dimension_ic", "D1")
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path, test_counter=counter)
            text = report_path.read_text()
            self.assertIn("検定数: 3", text)

    def test_all_sections_present(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path)
            text = report_path.read_text()
            for heading in ["0. 健全性", "1. 単変量", "2. 次元スコアIC", "3. 総合",
                            "4. ベースライン", "5. 多重検定", "6. 生存バイアス"]:
                self.assertIn(heading, text)

    def test_tolerates_a_zero_byte_csv_from_an_empty_table(self):
        # 回帰テスト: 列名が1つも無いDataFrameをto_csvすると0バイトのファイルになり、
        # 素朴なread_csvはEmptyDataErrorで落ちる。実データ(合成DRYRUN)で実際に踏んだ
        # 不具合の再現ケース
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            (out_dir / "0_sanity_distribution.csv").write_text("", encoding="utf-8")
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path)  # 例外を出さずに完走すること
            self.assertTrue(report_path.exists())

    def test_links_to_calibration_report_when_given(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            calib_path = Path(tmp) / "reports" / "calibration_nishimura.md"
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text("dummy", encoding="utf-8")
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path, calibration_report_path=calib_path)
            text = report_path.read_text()
            self.assertIn("calibration_nishimura.md", text)

    def test_index_source_label_is_stated_when_given(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(
                out_dir, report_path,
                index_source_label="日経225（TOPIX取得失敗のためフォールバック。"
                                   "D2相対力・地合いゲージは日経225基準）")
            text = report_path.read_text()
            self.assertIn("指数: 日経225（TOPIX取得失敗のためフォールバック", text)

    def test_index_source_label_omitted_when_not_given(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "l1"
            self._write_minimal_layer1_output(out_dir)
            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path)
            text = report_path.read_text()
            self.assertNotIn("- 指数:", text)


class LoadIndexSourceLabelTest(unittest.TestCase):
    """store/index_meta.json（cli.py step_index が書く）からレポート用の説明文を作る。"""

    def test_missing_file_returns_none(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(report._load_index_source_label(str(Path(tmp) / "does_not_exist.json")))

    def test_fallback_meta_mentions_topix_failure(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "index_meta.json"
            p.write_text('{"ticker": "^N225", "label": "日経225", "is_fallback": true}', encoding="utf-8")
            label = report._load_index_source_label(str(p))
            self.assertIn("日経225", label)
            self.assertIn("TOPIX取得失敗", label)

    def test_primary_meta_is_plain_label(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "index_meta.json"
            p.write_text('{"ticker": "^TPX", "label": "TOPIX", "is_fallback": false}', encoding="utf-8")
            self.assertEqual(report._load_index_source_label(str(p)), "TOPIX")


class EndToEndReportTest(unittest.TestCase):
    """run_layer1 の実出力 → build_l1_report が壊れずに一気通貫で動くことを確認。"""

    def test_real_layer1_output_can_be_bundled(self):
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
            out_dir = Path(tmp) / "l1"
            counter = layer1.run_layer1(pool, out_dir, ohlcv=ohlcv, listed=listed, h=10, n_draws=3,
                                        seed=0, nishimura_start=pd.Timestamp("2021-08-10"),
                                        nishimura_end=replay_end, log=lambda *a: None)

            calib_table = calibration.run_calibration(
                ohlcv, tickers, windows=[("smoke", pd.Timestamp("2021-08-10"), replay_end)])
            calib_path = calibration.write_calibration_report(
                calib_table, Path(tmp) / "reports" / "calibration_nishimura.md")

            report_path = Path(tmp) / "reports" / "l1_report.md"
            report.build_l1_report(out_dir, report_path, test_counter=counter,
                                   window_label="smoke", calibration_report_path=calib_path)
            self.assertTrue(report_path.exists())
            text = report_path.read_text()
            self.assertIn(f"検定数: {counter.count}", text)
            self.assertIn("calibration_nishimura.md", text)


if __name__ == "__main__":
    unittest.main()
