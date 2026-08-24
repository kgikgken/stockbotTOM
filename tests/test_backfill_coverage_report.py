"""scripts/backfill_coverage_report.py の分類ロジックのテスト（診断用スクリプト）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_coverage_report import build_report, classify_row  # noqa: E402


class ClassifyRowTest(unittest.TestCase):
    def test_a_full_history(self):
        row = pd.Series({"fetched": True, "bars": 2400, "first_date": pd.Timestamp("2010-01-01")})
        self.assertEqual(classify_row(row), "A")

    def test_b_new_listing_short_history_is_normal(self):
        row = pd.Series({"fetched": True, "bars": 100, "first_date": pd.Timestamp("2026-01-01")})
        self.assertEqual(classify_row(row), "B")

    def test_c_old_listing_but_short_history_is_suspicious(self):
        row = pd.Series({"fetched": True, "bars": 500, "first_date": pd.Timestamp("2010-01-01")})
        self.assertEqual(classify_row(row), "C")

    def test_c_fetch_failed(self):
        row = pd.Series({"fetched": False, "bars": 0, "first_date": pd.NaT})
        self.assertEqual(classify_row(row), "C")


class BuildReportTest(unittest.TestCase):
    def test_end_to_end_classification(self):
        from stockbot.data.synthetic import make_synthetic

        end = pd.Timestamp("2026-08-21")
        ohlcv = {}
        ohlcv.update(make_synthetic(["1000.T"], n_bars=2500, end=end))  # A
        ohlcv.update(make_synthetic(["1001.T"], n_bars=100, end=end))   # B (新規上場)
        # 1002.T: 取得失敗（ohlcv に含めない）

        listed = pd.DataFrame({
            "ticker": ["1000.T", "1001.T", "1002.T"],
            "code": ["1000", "1001", "1002"],
            "name": ["A", "B", "C"],
            "market": ["プライム"] * 3,
            "sector33": ["x"] * 3,
            "is_equity": [True] * 3,
        })

        u = build_report(ohlcv, listed)
        cls = dict(zip(u["ticker"], u["class"]))
        self.assertEqual(cls["1000.T"], "A")
        self.assertEqual(cls["1001.T"], "B")
        self.assertEqual(cls["1002.T"], "C")


if __name__ == "__main__":
    unittest.main()
