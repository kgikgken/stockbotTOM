"""日次特徴量の保存（T-206/T-208/T-301）。列名の安定性・状態/ゲートフィルタ・保存往復・
プール読み込みを確認する。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.features import dimensions, gates, pullback
from stockbot.pipeline import (
    BOOL_COLS,
    DAILY_FEATURES_COLS,
    DIMENSION_SCORE_COLS,
    SCORE_COLS,
    STRUCT_COLS,
    compute_daily_features,
    load_recent_daily_features,
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
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        self.assertEqual(list(df.columns), DAILY_FEATURES_COLS)
        for fid in dimensions.FEATURE_IDS:
            self.assertIn(fid, df.columns)
        for col in gates.GATE_COLS:
            self.assertIn(col, df.columns)
        for col in DIMENSION_SCORE_COLS + SCORE_COLS:
            self.assertIn(col, df.columns)

    def test_kept_rows_all_pass_all_gates(self):
        """採点対象として残った行は全てゲートを通過しているはず（T-208）。"""
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        if len(df):
            self.assertTrue(df["gate_pass"].all())
            self.assertTrue((df[["g0", "g1", "g2", "g3"]] == True).all().all())  # noqa: E712

    def test_score_columns_are_populated_without_history_pool(self):
        """history_pool 無しでも当日ぶんだけをプールとして次元・総合スコアが埋まる（T-301）。"""
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        if len(df):
            for col in DIMENSION_SCORE_COLS:
                self.assertTrue(df[col].between(0, 1).all(), msg=col)
            for col in SCORE_COLS:
                self.assertTrue(df[col].between(0, 100).all(), msg=col)

    def test_empty_result_still_has_full_schema(self):
        empty = compute_daily_features({}, [], pd.Series([100.0, 101.0],
                                        index=pd.bdate_range("2024-01-01", periods=2)), K, label_n=15)
        self.assertEqual(list(empty.columns), DAILY_FEATURES_COLS)
        self.assertEqual(len(empty), 0)


class ScorableStateFilterTest(unittest.TestCase):
    def test_only_forming_bounce_break_states_are_kept(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        allowed = {pullback.STATE_FORMING, pullback.STATE_BOUNCE, pullback.STATE_BREAK}
        self.assertTrue(set(df["state"].unique()).issubset(allowed))
        self.assertGreater(len(df), 0, "合成データなら少なくとも1件は採点対象になるはず")

    def test_short_history_ticker_is_skipped_without_crashing(self):
        tickers, ohlcv, idx_close = _synthetic()
        short_ticker = "9999.T"
        ohlcv[short_ticker] = ohlcv[tickers[0]].iloc[:10].copy()
        df = compute_daily_features(ohlcv, tickers + [short_ticker], idx_close, K, label_n=15)
        self.assertNotIn(short_ticker, set(df["ticker"]))


class RegimeWiringTest(unittest.TestCase):
    def test_regime_and_breadth_columns_are_populated(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
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
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
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


class HistoryPoolTest(unittest.TestCase):
    """load_recent_daily_features とプール連結（T-301）。"""

    def test_load_recent_excludes_asof_and_returns_bool_dtype(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        self.assertGreater(len(df), 0)
        with TemporaryDirectory() as tmp:
            # save_daily_features のファイル名は asof 引数由来、中身の date 列は別物なので
            # スナップショットごとに date 列を明示的にずらして「別日」を模す
            for day in ("2026-08-19", "2026-08-20", "2026-08-21"):
                snap = df.copy()
                snap["date"] = pd.Timestamp(day)
                save_daily_features(snap, Path(tmp), pd.Timestamp(day))

            pool = load_recent_daily_features(Path(tmp), pd.Timestamp("2026-08-21"), pool_days=20)
            # asof (08-21) 自身は含まれない。08-19/08-20 の2日ぶんだけ
            self.assertEqual(sorted(pool["date"].unique()),
                             sorted(pd.to_datetime(["2026-08-19", "2026-08-20"])))
            for col in BOOL_COLS:
                self.assertTrue(pd.api.types.is_bool_dtype(pool[col]), msg=col)
                # 文字列 "False" が真扱いになっていないこと
                self.assertTrue(((pool[col] == False) | (pool[col] == True) | pool[col].isna()).all())  # noqa: E712

    def test_load_recent_caps_at_pool_days_minus_one(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        with TemporaryDirectory() as tmp:
            days = pd.bdate_range("2026-08-01", periods=11)
            for day in days[:10]:
                snap = df.copy()
                snap["date"] = day
                save_daily_features(snap, Path(tmp), day)
            asof = days[10]  # 保存した10日すべてより後
            pool = load_recent_daily_features(Path(tmp), asof, pool_days=5)
            self.assertEqual(pool["date"].nunique(), 4)

    def test_compute_daily_features_accepts_history_pool_without_crashing(self):
        tickers, ohlcv, idx_close = _synthetic()
        df = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15)
        with TemporaryDirectory() as tmp:
            snap = df.copy()
            snap["date"] = pd.Timestamp("2026-08-20")
            save_daily_features(snap, Path(tmp), pd.Timestamp("2026-08-20"))
            pool = load_recent_daily_features(Path(tmp), pd.Timestamp("2026-08-21"), pool_days=20)
            df2 = compute_daily_features(ohlcv, tickers, idx_close, K, label_n=15,
                                         history_pool=pool, pool_days=20)
            self.assertEqual(list(df2.columns), DAILY_FEATURES_COLS)
            if len(df2):
                for col in DIMENSION_SCORE_COLS:
                    self.assertTrue(df2[col].between(0, 1).all(), msg=col)


def _acceptance_ohlcv(ticker="1301.T"):
    """test_dimensions.py の _build_acceptance_series と同じ合成 pullback 系列
    （形成中→反発開始→ブレイク→完了）を、compute_daily_features に渡せる
    OHLCV DataFrame の形にしたもの（ゲート通過・採点対象になることが保証された
    決定論的な系列。make_synthetic はランダムでゲート通過が保証されないため、
    位置整合バグの回帰テストにはこちらを使う）。
    """
    warm_end = 180.0
    warmup = np.linspace(100.0, warm_end, 220)
    dip = np.array([180, 178, 176, 174, 176, 178, 180.0]) + (warmup[-1] - 180)
    rise = np.linspace(dip[-1], 220.0, 15)[1:]
    peak = np.array([220, 222, 224, 230, 224, 222, 220.0])
    close = np.concatenate([warmup, dip, rise, peak])
    pullback_forming = np.array([217, 214, 211, 209])
    bounce_trigger_day = np.array([214])
    post_bounce = np.array([217, 220])
    breakout = np.array([232])
    close_full = np.concatenate(
        [close, pullback_forming, bounce_trigger_day, post_bounce, breakout])
    # test_dimensions.py で t_pos=253 が STATE_BREAK（ゲート通過しうる採点対象状態）と
    # 確認済みの日。compute_daily_features は各銘柄の「最終日」だけを評価する
    # （t_pos = len(df)-1）ため、系列をその日までで打ち切って最終日をそこに合わせる
    close_full = close_full[:254]

    idx = pd.bdate_range("2024-01-01", periods=len(close_full))
    close_s = pd.Series(close_full, dtype=float, index=idx)
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "Open": close_s - 0.1, "High": close_s + 0.3, "Low": close_s - 0.3,
        "Close": close_s, "Volume": rng.uniform(5e5, 2e6, len(close_full)),
        "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx)
    full_idx_close = pd.Series(500 + np.cumsum(rng.normal(0, 1, len(close_full))), index=idx)
    return {ticker: df}, full_idx_close


class IndexAlignmentTest(unittest.TestCase):
    """回帰テスト: 個別銘柄の日付集合と指数(idx_close)の日付集合が完全には一致しない
    場合でも compute_daily_features がクラッシュしないこと。dimensions.compute_dimensions
    は idx_close が close と同じ位置規約で整列済みという前提で t_pos をそのまま使い回す
    （D2の計算）。銘柄ごとに実際に取得できた営業日が指数と食い違うと、位置だけを
    信用した参照は日付の取り違えや IndexError を起こす
    （2026-08-24、10年replayの実データで発生: 指数の方が銘柄より短い履歴しか無く
    IndexError: index out of bounds になった）。pipeline.compute_daily_features が
    呼び出し側で idx_close を close.index に日付整列してから渡すことで防ぐ。
    """

    def test_index_shorter_than_ticker_history_does_not_crash(self):
        ohlcv, full_idx_close = _acceptance_ohlcv()
        ticker = next(iter(ohlcv))
        # 指数側だけ先頭を欠かせる（銘柄は全期間、指数はそれより短い期間しか
        # 取得できていない状態を模す）
        short_idx_close = full_idx_close.iloc[50:]
        self.assertLess(len(short_idx_close), len(ohlcv[ticker]))

        out = compute_daily_features(ohlcv, [ticker], short_idx_close, k=K, label_n=15,
                                     log=lambda *a: None)
        self.assertGreater(len(out), 0, "決定論的な受入系列は採点対象になるはず")

    def test_d2_uses_date_matched_index_values_not_raw_position(self):
        ohlcv, full_idx_close = _acceptance_ohlcv()
        ticker = next(iter(ohlcv))
        short_idx_close = full_idx_close.iloc[50:]

        out = compute_daily_features(ohlcv, [ticker], short_idx_close, k=K, label_n=15,
                                     log=lambda *a: None)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]

        close = ohlcv[ticker]["Close"]
        t_pos = len(close) - 1
        aligned = short_idx_close.reindex(close.index)  # 期待値: 日付で整列した値
        for n, fid in ((60, "d2_rs60"), (120, "d2_rs120")):
            i0, i1 = aligned.iloc[t_pos - n], aligned.iloc[t_pos]
            if pd.isna(i0) or pd.isna(i1):
                self.assertTrue(pd.isna(row[fid]), msg=fid)
            else:
                c0, c1 = close.iloc[t_pos - n], close.iloc[t_pos]
                expected = float(np.log(c1 / c0) - np.log(i1 / i0))
                self.assertAlmostEqual(row[fid], expected, places=8, msg=fid)


if __name__ == "__main__":
    unittest.main()
