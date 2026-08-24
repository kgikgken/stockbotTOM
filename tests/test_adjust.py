import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.adjust import check_splits


def _series(n=60, split_at=30, ratio=2.0, adjusted=True, record=True):
    idx = pd.bdate_range(end="2026-08-21", periods=n)
    close = np.full(n, 2000.0) + np.random.default_rng(0).normal(0, 5, n)
    vol = np.full(n, 1e5)
    if not adjusted:           # 未調整: 分割前の価格が比率倍で高い
        close[:split_at] *= ratio
        vol[:split_at] /= ratio
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": vol, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    if record:
        df.iloc[split_at, df.columns.get_loc("Stock Splits")] = ratio
    return df


class SplitCheckTest(unittest.TestCase):
    def test_adjusted_series_untouched(self):
        df = _series(adjusted=True, record=True)
        fixed, issues = check_splits(df, "X")
        self.assertEqual(issues, [])
        self.assertTrue(np.allclose(fixed["Close"], df["Close"]))

    def test_unadjusted_recorded_split_is_repaired(self):
        df = _series(adjusted=False, record=True)
        fixed, issues = check_splits(df, "X")
        self.assertEqual([i["kind"] for i in issues], ["unadjusted_split"])
        ratio = fixed["Close"].iloc[29] / fixed["Close"].iloc[30]
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)
        self.assertAlmostEqual(fixed["Volume"].iloc[0], 1e5, delta=1)

    def test_unrecorded_jump_is_flagged_not_fixed(self):
        df = _series(adjusted=False, record=False)
        fixed, issues = check_splits(df, "X")
        kinds = [i["kind"] for i in issues]
        self.assertIn("suspected_unrecorded_split", kinds)
        self.assertTrue(np.allclose(fixed["Close"], df["Close"]))

    def test_int64_volume_does_not_raise_lossy_setitem_error(self):
        # 回帰テスト: yfinance の実データは Volume が int64 で返る。未調整分割の出来高
        # 修正（比率倍）で LossySetitemError になっていた不具合（backfill ワークフローの
        # 実データ取得で実際に踏んだ）。テスト用の合成 Volume は float なので、この
        # dtype を明示的に int64 にしてから通す
        df = _series(adjusted=False, record=True)
        df["Volume"] = df["Volume"].astype("int64")
        fixed, issues = check_splits(df, "X")  # 例外を出さずに完走すること
        self.assertEqual([i["kind"] for i in issues], ["unadjusted_split"])
        self.assertAlmostEqual(fixed["Volume"].iloc[0], 1e5, delta=1)


if __name__ == "__main__":
    unittest.main()
