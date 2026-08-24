"""scripts/replay_run_summary.py の集計ロジックのテスト（診断用スクリプト）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_run_summary import build_summary  # noqa: E402


def _table(rows):
    return pd.DataFrame(rows)


class BuildSummaryTest(unittest.TestCase):
    def test_empty_table(self):
        summary = build_summary(pd.DataFrame(columns=["date", "ticker", "d2_rs60", "d2_rs120"]))
        self.assertEqual(summary["total_rows"], 0)
        self.assertEqual(summary["n_days_with_rows"], 0)
        self.assertIsNone(summary["avg_scored_per_day"])

    def test_counts_rows_and_days_and_missing_rate(self):
        table = _table([
            {"date": "2021-08-02", "ticker": "1301.T", "d2_rs60": 0.01, "d2_rs120": np.nan},
            {"date": "2021-08-02", "ticker": "1302.T", "d2_rs60": np.nan, "d2_rs120": 0.02},
            {"date": "2021-08-03", "ticker": "1301.T", "d2_rs60": 0.03, "d2_rs120": 0.04},
        ])
        summary = build_summary(table)
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["n_days_with_rows"], 2)
        self.assertAlmostEqual(summary["avg_scored_per_day"], 1.5)
        self.assertAlmostEqual(summary["d2_rs60_missing_rate"], 1 / 3)
        self.assertAlmostEqual(summary["d2_rs120_missing_rate"], 1 / 3)

    def test_missing_rate_is_none_when_column_absent(self):
        table = _table([{"date": "2021-08-02", "ticker": "1301.T"}])
        summary = build_summary(table)
        self.assertIsNone(summary["d2_rs60_missing_rate"])
        self.assertIsNone(summary["d2_rs120_missing_rate"])


if __name__ == "__main__":
    unittest.main()
