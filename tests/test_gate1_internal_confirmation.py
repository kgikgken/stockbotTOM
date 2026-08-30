"""scripts/gate1_internal_confirmation.py の窓フィルタのテスト（2026-08-30 指摘への
対応: data/a_prime は日付ベースのキャッシュのため設計窓(2021-08-01〜2024-01-31)と
内部確認窓(2024-02-01〜2026-01-30)の出力が同一キャッシュに蓄積されうる。集計前に
窓フィルタが両方の入力（(c) pool と (a′)）に効いているかを確認する）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

from . import _path  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gate1_internal_confirmation import (  # noqa: E402
    INTERNAL_CONFIRMATION_WINDOW,
    _filter_window,
)


class FilterWindowTest(unittest.TestCase):
    def test_excludes_design_window_rows_mixed_in_same_frame(self):
        # data/a_prime のキャッシュが両窓を蓄積している状況を模す
        df = pd.DataFrame({
            "ticker": ["T1"] * 4,
            "date": [
                pd.Timestamp("2021-08-02"),   # 設計窓
                pd.Timestamp("2024-01-31"),   # 設計窓の最終日
                pd.Timestamp("2024-02-01"),   # 内部確認窓の初日
                pd.Timestamp("2026-01-30"),   # 内部確認窓の最終日
            ],
            "r_3": [0.01, 0.02, 0.03, 0.04],
        })
        start, end = INTERNAL_CONFIRMATION_WINDOW
        out = _filter_window(df, start, end)
        self.assertEqual(sorted(out["date"].dt.date.astype(str)),
                         ["2024-02-01", "2026-01-30"])

    def test_excludes_dates_after_window(self):
        df = pd.DataFrame({
            "ticker": ["T1"] * 2,
            "date": [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-02")],
            "r_3": [0.01, 0.02],
        })
        start, end = INTERNAL_CONFIRMATION_WINDOW
        out = _filter_window(df, start, end)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["date"], pd.Timestamp("2026-01-30"))

    def test_empty_frame_passthrough(self):
        df = pd.DataFrame(columns=["ticker", "date", "r_3"])
        start, end = INTERNAL_CONFIRMATION_WINDOW
        out = _filter_window(df, start, end)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
