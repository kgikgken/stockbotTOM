"""西村ルール較正（DESIGN.md §10.2(b) / TASKS.md T-404）のテスト。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.validation import calibration
from stockbot.validation.replay import HOLDOUT_WINDOW


def _series(closes: list[float], start: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=idx)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close}, index=idx)


def _one_trade_series(start: str) -> pd.DataFrame:
    """test_layer1.NishimuraTradesTest と同じ構成: 1トレード確実に発生する系列。"""
    up = list(np.linspace(70.0, 100.0, 80))
    dip = [96.0, 90.0]
    recover = [92.0, 101.0, 102.0]
    closes = up + dip + recover + [102.0] * 15
    return _series(closes, start)


class RunCalibrationTest(unittest.TestCase):
    def test_source_row_is_first_and_has_no_diff(self):
        table = calibration.run_calibration({}, [], windows=[])
        self.assertEqual(table.iloc[0]["window"], calibration.SOURCE_LABEL)
        self.assertEqual(table.iloc[0]["win_rate"], calibration.SOURCE_WIN_RATE)
        self.assertTrue(np.isnan(table.iloc[0]["win_rate_diff"]))

    def test_diff_columns_are_ours_minus_source(self):
        df = _one_trade_series("2016-01-04")
        windows = [("test_window", pd.Timestamp("2016-01-04"), pd.Timestamp("2016-12-30"))]
        table = calibration.run_calibration({"1301.T": df}, ["1301.T"], windows=windows)
        row = table[table["window"] == "test_window"].iloc[0]
        self.assertEqual(row["n_trades"], 1)
        self.assertAlmostEqual(row["win_rate_diff"], row["win_rate"] - calibration.SOURCE_WIN_RATE, places=9)
        self.assertAlmostEqual(row["avg_return_diff"],
                               row["avg_return"] - calibration.SOURCE_AVG_RETURN, places=9)
        # 唯一のトレードが勝ちトレード(損失0件)なのでPFは無限大。差分は意味を持たないためNaN
        self.assertTrue(np.isinf(row["profit_factor"]))
        self.assertTrue(np.isnan(row["profit_factor_diff"]))
        self.assertAlmostEqual(row["avg_hold_days_diff"],
                               row["avg_hold_days"] - calibration.SOURCE_AVG_HOLD_DAYS, places=9)

    def test_no_trades_gives_nan_diffs(self):
        # 一貫した下降トレンド: エントリー条件(close>sma75)を満たさない
        closes = list(np.linspace(150.0, 50.0, 100))
        df = _series(closes, "2016-01-04")
        windows = [("test_window", pd.Timestamp("2016-01-04"), pd.Timestamp("2016-12-30"))]
        table = calibration.run_calibration({"1301.T": df}, ["1301.T"], windows=windows)
        row = table[table["window"] == "test_window"].iloc[0]
        self.assertEqual(row["n_trades"], 0)
        self.assertTrue(np.isnan(row["win_rate_diff"]))


class HoldoutSafetyTest(unittest.TestCase):
    """CLAUDE.md 絶対規則: ホールドアウト（2026-02〜2026-08）を見ない。"""

    def test_data_after_holdout_start_is_truncated_before_simulation(self):
        # HOLDOUT_WINDOW[0]以降にしかエントリー条件を満たさない状況を作る
        pre = list(np.linspace(150.0, 50.0, 100))  # ホールドアウト前: 一貫下降(発火しない)
        idx_pre = pd.bdate_range("2025-06-01", periods=100)
        post_start = HOLDOUT_WINDOW[0] + pd.Timedelta(days=10)
        up = list(np.linspace(70.0, 100.0, 80))
        dip = [96.0, 90.0]
        recover = [92.0, 101.0, 102.0]
        post = up + dip + recover + [102.0] * 15  # ホールドアウト後: 発火する構成
        idx_post = pd.bdate_range(post_start, periods=len(post))
        close = pd.Series(pre + post, index=idx_pre.append(idx_post))
        df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                           "Close": close}, index=close.index)

        # windowの範囲自体はホールドアウトも含めて広く取る（内部で切るはずなので）
        windows = [("wide", idx_pre[0], idx_post[-1])]
        table = calibration.run_calibration({"1301.T": df}, ["1301.T"], windows=windows)
        row = table[table["window"] == "wide"].iloc[0]
        # ホールドアウト側のデータでしか発火しないシグナルなので、内部で切り捨てられ0件のはず
        self.assertEqual(row["n_trades"], 0)

    def test_truncate_before_holdout_drops_holdout_rows(self):
        idx = pd.bdate_range("2026-01-01", periods=60)
        close = pd.Series(np.arange(60.0) + 100, index=idx)
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close}, index=idx)
        out = calibration._truncate_before_holdout({"1301.T": df})
        self.assertTrue((out["1301.T"].index < HOLDOUT_WINDOW[0]).all())
        self.assertLess(len(out["1301.T"]), len(df))


class WriteCalibrationReportTest(unittest.TestCase):
    def test_writes_markdown_with_source_and_table(self):
        table = calibration.run_calibration({}, [], windows=[])
        with TemporaryDirectory() as tmp:
            path = calibration.write_calibration_report(table, Path(tmp) / "calibration_nishimura.md")
            self.assertTrue(path.exists())
            text = path.read_text()
            self.assertIn("西村ルール較正", text)
            self.assertIn(calibration.SOURCE_LABEL, text)
            self.assertIn("ホールドアウト", text)
            self.assertIn("| window |", text)


if __name__ == "__main__":
    unittest.main()
