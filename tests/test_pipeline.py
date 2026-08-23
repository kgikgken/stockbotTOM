"""日次特徴量の保存（T-206）。列名の安定性・状態フィルタ・保存往復を確認する。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.features import dimensions, pullback
from stockbot.pipeline import (
    DAILY_FEATURES_COLS,
    DIMENSION_SCORE_COLS,
    SCORE_COLS,
    STRUCT_COLS,
    compute_daily_features,
    save_daily_features,
)

K = 3


def _synthetic(n_tickers=10, n_bars=400, seed=0):
    tickers = [f"{1301 + i * 37:04d}.T" for i in range(n_tickers)]
    end = pd.Timestamp("2026-08-21")
    ohlcv = make_synthetic(tickers, n_bars=n_bars, seed=seed, end=end)
    idx_close = make_synthetic_index(n_bars=n_bars, seed=seed, end=end)["Close"]
    return tickers, ohlcv, idx_close


class SchemaStabilityTest(unittest.TestCase):
    def test_columns_are_stable_and_include_all_registered_features(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K)
        self.assertEqual(list(df.columns), DAILY_FEATURES_COLS)
        for fid in dimensions.FEATURE_IDS:
            self.assertIn(fid, df.columns)
        for col in DIMENSION_SCORE_COLS + SCORE_COLS:
            self.assertIn(col, df.columns)

    def test_score_columns_are_nan_placeholders(self):
        """T-301/T-302 未実装のため、次元スコア・総合スコアは常にNaN。"""
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K)
        if len(df):
            for col in DIMENSION_SCORE_COLS + SCORE_COLS:
                self.assertTrue(df[col].isna().all(), msg=col)

    def test_empty_result_still_has_full_schema(self):
        empty = compute_daily_features({}, [], pd.Series([100.0, 101.0],
                                        index=pd.bdate_range("2024-01-01", periods=2)), K)
        self.assertEqual(list(empty.columns), DAILY_FEATURES_COLS)
        self.assertEqual(len(empty), 0)


class ScorableStateFilterTest(unittest.TestCase):
    def test_only_forming_bounce_break_states_are_kept(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K)
        allowed = {pullback.STATE_FORMING, pullback.STATE_BOUNCE, pullback.STATE_BREAK}
        self.assertTrue(set(df["state"].unique()).issubset(allowed))
        self.assertGreater(len(df), 0, "合成データなら少なくとも1件は採点対象になるはず")

    def test_short_history_ticker_is_skipped_without_crashing(self):
        tickers, ohlcv, idx_close = _synthetic()
        short_ticker = "9999.T"
        ohlcv[short_ticker] = ohlcv[tickers[0]].iloc[:10].copy()
        df = compute_daily_features(ohlcv, tickers + [short_ticker], idx_close, K)
        self.assertNotIn(short_ticker, set(df["ticker"]))


class RegimeWiringTest(unittest.TestCase):
    def test_regime_and_breadth_columns_are_populated(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K)
        if len(df):
            self.assertTrue(df["regime"].isin(["強", "中", "弱"]).all())
            self.assertTrue(((df["breadth_75"] >= 0) & (df["breadth_75"] <= 1)).all())
            self.assertTrue(((df["breadth_200"] >= 0) & (df["breadth_200"] <= 1)).all())
            # 地合い・ブレスは日次で1回だけ計算し全銘柄で共有するので、値は全行で同一のはず
            self.assertEqual(df["regime"].nunique(), 1)
            self.assertEqual(df["breadth_75"].nunique(), 1)


class SaveDailyFeaturesTest(unittest.TestCase):
    def test_filename_and_roundtrip(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K)
        with TemporaryDirectory() as tmp:
            path = save_daily_features(df, Path(tmp), pd.Timestamp("2026-08-21"))
            self.assertEqual(path.name, "features_2026-08-21.csv.gz")
            self.assertTrue(path.exists())
            back = pd.read_csv(path)
            self.assertEqual(list(back.columns), DAILY_FEATURES_COLS)
            self.assertEqual(len(back), len(df))

    def test_struct_cols_cover_pullback_fields(self):
        expected = {"ticker", "date", "state", "h0_date", "l0_date", "lp_date",
                   "h0_high", "l0_low", "lp_value", "r", "leg", "leg_bars", "d",
                   "depth_pct", "depth_atr", "retrace", "position", "dev5",
                   "is_shallow", "is_deep"}
        self.assertEqual(set(STRUCT_COLS), expected)


if __name__ == "__main__":
    unittest.main()
