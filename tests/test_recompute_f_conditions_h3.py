"""scripts/recompute_f_conditions_h3.py の集計ロジックのテスト（診断用スクリプト。
2026-08-29 指示: hをh=3に凍結したため、F1〜F14の境界値をh=3で再チェックする）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

from . import _path  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recompute_f_conditions_h3 import recompute_f_conditions, summarize  # noqa: E402


class RecomputeFConditionsTest(unittest.TestCase):
    def test_still_negative_tail_flagged_true(self):
        pool = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(1, 11)],
            "date": [pd.Timestamp("2024-01-01")] * 10,
            "d2_rs60": [float(i) for i in range(1, 11)],
            "r_3": [-0.01] * 9 + [-0.08],  # T10(最大値=末端)のr_3が負のまま
        })
        table = recompute_f_conditions(pool, h=3)
        row = table[table["fid"] == "F10"].iloc[0]
        self.assertEqual(row["n_excluded"], 1)  # d2_rs60=10のみpctl>0.90で除外
        self.assertAlmostEqual(row["mean_r_3_tail"], -0.08, places=9)
        self.assertTrue(row["still_negative"] is True)

    def test_flipped_tail_flagged_false(self):
        pool = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(1, 11)],
            "date": [pd.Timestamp("2024-01-01")] * 10,
            "d2_rs60": [float(i) for i in range(1, 11)],
            "r_3": [-0.01] * 9 + [0.05],  # T10のr_3が正に転じている
        })
        table = recompute_f_conditions(pool, h=3)
        row = table[table["fid"] == "F10"].iloc[0]
        self.assertAlmostEqual(row["mean_r_3_tail"], 0.05, places=9)
        self.assertTrue(row["still_negative"] is False)

    def test_missing_feature_gives_undetermined(self):
        pool = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(1, 11)],
            "date": [pd.Timestamp("2024-01-01")] * 10,
            "d2_rs60": [float(i) for i in range(1, 11)],
            "r_3": [-0.01] * 10,
        })
        table = recompute_f_conditions(pool, h=3)
        # d3_depth_pct列が無いのでF1・F2は判定不能
        for fid in ("F1", "F2"):
            row = table[table["fid"] == fid].iloc[0]
            self.assertEqual(row["n_excluded"], 0)
            self.assertIsNone(row["still_negative"])

    def test_summarize_counts_match(self):
        pool = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(1, 11)],
            "date": [pd.Timestamp("2024-01-01")] * 10,
            "d2_rs60": [float(i) for i in range(1, 11)],
            "r_3": [-0.01] * 9 + [0.05],
        })
        table = recompute_f_conditions(pool, h=3)
        s = summarize(table)
        self.assertEqual(s["n_total"], 14)
        self.assertEqual(s["n_still_negative"] + s["n_flipped"] + s["n_undetermined"], 14)
        self.assertEqual(s["n_flipped"], 1)  # F10のみ該当あり、かつ反転


if __name__ == "__main__":
    unittest.main()
