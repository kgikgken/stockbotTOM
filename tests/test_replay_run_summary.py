"""scripts/replay_run_summary.py の集計ロジックのテスト（診断用スクリプト）。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_run_summary import build_summary, by_year_summary, empty_file_breakdown  # noqa: E402


def _table(rows):
    return pd.DataFrame(rows)


class BuildSummaryTest(unittest.TestCase):
    def test_empty_table(self):
        summary = build_summary(pd.DataFrame(columns=["date", "ticker", "d2_rs60", "d2_rs120"]))
        self.assertEqual(summary["total_rows"], 0)
        self.assertEqual(summary["n_days_with_rows"], 0)
        self.assertIsNone(summary["avg_scored_per_day"])

    def test_counts_rows_and_days_and_f_missing_rates(self):
        table = _table([
            {"date": "2021-08-02", "ticker": "1301.T", "d2_rs60": 0.01, "d2_rs120": np.nan},
            {"date": "2021-08-02", "ticker": "1302.T", "d2_rs60": np.nan, "d2_rs120": 0.02},
            {"date": "2021-08-03", "ticker": "1301.T", "d2_rs60": 0.03, "d2_rs120": 0.04},
        ])
        summary = build_summary(table)
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["n_days_with_rows"], 2)
        self.assertAlmostEqual(summary["avg_scored_per_day"], 1.5)
        self.assertAlmostEqual(summary["f_missing_rates"]["F10"]["missing_rate"], 1 / 3)  # d2_rs60
        self.assertEqual(summary["f_missing_rates"]["F10"]["column"], "d2_rs60")
        self.assertAlmostEqual(summary["f_missing_rates"]["F11"]["missing_rate"], 1 / 3)  # d2_rs120

    def test_missing_rate_is_none_when_column_absent(self):
        table = _table([{"date": "2021-08-02", "ticker": "1301.T"}])
        summary = build_summary(table)
        self.assertIsNone(summary["f_missing_rates"]["F10"]["missing_rate"])

    def test_warmup_rows_excluded_by_default(self):
        table = _table([
            {"date": "2017-02-13", "ticker": "1301.T", "d2_rs60": 0.01, "warmup": True},
            {"date": "2017-03-15", "ticker": "1301.T", "d2_rs60": 0.02, "warmup": False},
        ])
        summary = build_summary(table)
        self.assertEqual(summary["total_rows"], 1)
        self.assertEqual(summary["n_warmup_rows_excluded"], 1)

    def test_warmup_rows_included_when_requested(self):
        table = _table([
            {"date": "2017-02-13", "ticker": "1301.T", "d2_rs60": 0.01, "warmup": True},
            {"date": "2017-03-15", "ticker": "1301.T", "d2_rs60": 0.02, "warmup": False},
        ])
        summary = build_summary(table, exclude_warmup=False)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["n_warmup_rows_excluded"], 0)


class ByYearSummaryTest(unittest.TestCase):
    """2026-08-29指示: 頑健性窓の年別1日あたり行数（記録用）。"""

    def test_groups_by_year_and_averages_per_day(self):
        table = _table([
            {"date": "2018-06-01", "ticker": "1301.T"},
            {"date": "2018-06-01", "ticker": "1302.T"},
            {"date": "2018-06-04", "ticker": "1301.T"},
            {"date": "2019-01-07", "ticker": "1301.T"},
        ])
        out = by_year_summary(table)
        self.assertEqual(list(out["year"]), [2018, 2019])
        self.assertEqual(out.iloc[0]["n_days"], 2)
        self.assertEqual(out.iloc[0]["total_rows"], 3)
        self.assertAlmostEqual(out.iloc[0]["avg_rows_per_day"], 1.5)
        self.assertEqual(out.iloc[1]["n_days"], 1)

    def test_excludes_warmup_rows_by_default(self):
        table = _table([
            {"date": "2017-02-13", "ticker": "1301.T", "warmup": True},
            {"date": "2017-03-15", "ticker": "1301.T", "warmup": False},
        ])
        out = by_year_summary(table)
        self.assertEqual(list(out["year"]), [2017])
        self.assertEqual(out.iloc[0]["total_rows"], 1)

    def test_empty_table_gives_empty_frame_with_schema(self):
        out = by_year_summary(pd.DataFrame(columns=["date", "ticker"]))
        self.assertEqual(list(out.columns), ["year", "n_days", "total_rows", "avg_rows_per_day"])
        self.assertEqual(len(out), 0)


class EmptyFileBreakdownTest(unittest.TestCase):
    def test_classifies_by_sidecar_meta_and_defaults_missing_meta_to_no_data(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            import gzip
            # 候補0件（サイドカーあり）
            with gzip.open(tmp / "replay_2020-03-02.csv.gz", "wt") as f:
                f.write("ticker\n")
            (tmp / "replay_meta_2020-03-02.json").write_text(
                json.dumps({"empty_reason": "no_candidates"}), encoding="utf-8")

            # データ無し（サイドカーあり）
            with gzip.open(tmp / "replay_2020-03-03.csv.gz", "wt") as f:
                f.write("ticker\n")
            (tmp / "replay_meta_2020-03-03.json").write_text(
                json.dumps({"empty_reason": "no_data"}), encoding="utf-8")

            # サイドカーが無い空ファイル（このガード追加前の旧データを想定）→ 安全側でデータ無し扱い
            with gzip.open(tmp / "replay_2020-03-04.csv.gz", "wt") as f:
                f.write("ticker\n")

            breakdown = empty_file_breakdown(tmp)
            self.assertEqual(breakdown, {"no_data": 2, "no_candidates": 1})


if __name__ == "__main__":
    unittest.main()
