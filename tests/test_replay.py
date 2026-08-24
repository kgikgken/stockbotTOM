"""歴史的再生（DESIGN.md §10.1・§10.5 / TASKS.md T-402）のテスト。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.features import indicators
from stockbot.pipeline import DAILY_FEATURES_COLS, compute_daily_features
from stockbot.validation import labels, replay

N_BARS = 320
DATE_T = pd.Timestamp("2026-06-19")
TICKERS = [f"{1301 + i * 37:04d}.T" for i in range(30)]


def _listed(tickers=TICKERS) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "is_equity": True})


def _extend(df: pd.DataFrame, n_extra: int, seed: int) -> pd.DataFrame:
    """系列の末尾から続く「未来」のダミー日足を足す（ラベル計算用の先読みデータ）。"""
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


def _assert_features_match(a: pd.DataFrame, b: pd.DataFrame) -> None:
    a = a.sort_values("ticker").reset_index(drop=True)
    b = b.sort_values("ticker").reset_index(drop=True)
    assert list(a["ticker"]) == list(b["ticker"]), (list(a["ticker"]), list(b["ticker"]))
    for col in DAILY_FEATURES_COLS:
        for i in range(len(a)):
            va, vb = a.loc[i, col], b.loc[i, col]
            if isinstance(va, bool) or isinstance(vb, bool):
                assert va == vb, (col, i, va, vb)
            elif pd.isna(va):
                assert pd.isna(vb), (col, i, va, vb)
            elif isinstance(va, str):
                assert va == vb, (col, i, va, vb)
            elif isinstance(va, pd.Timestamp):
                assert va == vb, (col, i, va, vb)
            else:
                assert abs(float(va) - float(vb)) < 1e-9, (col, i, va, vb)


class EquivalenceTest(unittest.TestCase):
    """必須テスト: 再生の1日分 = 日次パイプラインを同じTで走らせた結果（同一性）。"""

    def setUp(self):
        self.ohlcv_upto_t = make_synthetic(TICKERS, n_bars=N_BARS, seed=0, end=DATE_T)
        self.idx_upto_t = make_synthetic_index(N_BARS, seed=0, end=DATE_T)
        self.listed = _listed()
        # ticker文字列のhash()はプロセスごとに乱数化されるため種に使わない（再現性が壊れる）。
        # 銘柄コード（先頭4桁）を種にする
        self.ohlcv_full = {t: _extend(self.ohlcv_upto_t[t], 30, int(t[:4]))
                           for t in TICKERS}

    def test_replay_one_day_matches_direct_pipeline_call(self):
        replay_df = replay.replay_one_day(DATE_T, self.ohlcv_full, self.idx_upto_t, self.listed,
                                          k=3, label_n=15, pool_days=1, log=lambda *a: None)
        self.assertGreater(len(replay_df), 0, "合成データなら少なくとも1件は採点対象になるはず")

        universe = replay.replay_universe_tickers(self.listed, self.ohlcv_upto_t, DATE_T)
        direct_df = compute_daily_features(self.ohlcv_upto_t, universe, self.idx_upto_t["Close"],
                                           k=3, label_n=15, pool_days=1, log=lambda *a: None)
        _assert_features_match(replay_df, direct_df)

    def test_labels_match_direct_labels_call(self):
        replay_df = replay.replay_one_day(DATE_T, self.ohlcv_full, self.idx_upto_t, self.listed,
                                          k=3, label_n=15, pool_days=1, log=lambda *a: None)
        universe = replay.replay_universe_tickers(self.listed, self.ohlcv_upto_t, DATE_T)
        benchmarks_raw = labels.universe_benchmark_returns(
            {t: self.ohlcv_full[t] for t in universe}, DATE_T, labels.H_LIST)
        benchmarks = {h: v["mean"] for h, v in benchmarks_raw.items()}

        for _, row in replay_df.iterrows():
            ticker = row["ticker"]
            df_full = self.ohlcv_full[ticker]
            pos = replay._date_position(self.ohlcv_upto_t[ticker].index, DATE_T)
            df_trunc = df_full.iloc[: pos + 1]
            atr_t = float(indicators.atr_wilder(
                df_trunc["High"], df_trunc["Low"], df_trunc["Close"], 14).iloc[-1])
            expected = labels.compute_labels(
                df_full["Close"], df_full["Open"], df_full["High"], df_full["Low"],
                pos, row["h0_high"], row["lp_value"], atr_t, benchmarks, n=15)
            for key, val in expected.items():
                actual = row[key]
                # hit_day は未決なら None（compute_labels の戻り値）だが、DataFrame を
                # 経由すると NaN になる（int/None混在列はfloat化されるため）。どちらも
                # 「無い」として扱う
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    self.assertTrue(pd.isna(actual), msg=f"{ticker} {key}")
                else:
                    self.assertEqual(actual, val, msg=f"{ticker} {key}")


class UniverseFilterTest(unittest.TestCase):
    """DESIGN.md §10.1: ユニバース = 現在の上場銘柄のうちT時点で履歴250本以上ある銘柄。"""

    def test_ticker_with_enough_history_is_included_and_short_one_excluded(self):
        listed = _listed(["1301.T", "1302.T"])
        long_df = make_synthetic(["1301.T"], n_bars=300, seed=0, end=DATE_T)["1301.T"]
        short_df = long_df.iloc[-100:]  # 100本しか無い
        ohlcv = {"1301.T": long_df, "1302.T": short_df}
        universe = replay.replay_universe_tickers(listed, ohlcv, DATE_T, min_history_bars=250)
        self.assertIn("1301.T", universe)
        self.assertNotIn("1302.T", universe)

    def test_ticker_without_date_t_is_excluded(self):
        listed = _listed(["1301.T", "1302.T"])
        a = make_synthetic(["1301.T"], n_bars=300, seed=0, end=DATE_T)["1301.T"]
        b = make_synthetic(["1302.T"], n_bars=300, seed=0, end=pd.Timestamp("2020-01-01"))["1302.T"]
        universe = replay.replay_universe_tickers(listed, {"1301.T": a, "1302.T": b}, DATE_T,
                                                   min_history_bars=250)
        self.assertIn("1301.T", universe)
        self.assertNotIn("1302.T", universe)

    def test_non_equity_listing_is_excluded(self):
        listed = pd.DataFrame({"ticker": ["1301.T", "1302.T"], "is_equity": [True, False]})
        df = make_synthetic(["1301.T", "1302.T"], n_bars=300, seed=0, end=DATE_T)
        universe = replay.replay_universe_tickers(listed, df, DATE_T, min_history_bars=250)
        self.assertIn("1301.T", universe)
        self.assertNotIn("1302.T", universe)


class HoldoutFilterTest(unittest.TestCase):
    """CLAUDE.md 絶対規則: ホールドアウトは明示フラグ無しで生成しない。"""

    def test_holdout_dates_excluded_by_default(self):
        dates = pd.bdate_range("2026-01-26", "2026-02-06")
        kept = replay._filter_holdout(dates, include_holdout=False, log=lambda *a: None)
        self.assertTrue((kept < replay.HOLDOUT_WINDOW[0]).all())
        self.assertLess(len(kept), len(dates))

    def test_holdout_dates_kept_when_flag_true(self):
        dates = pd.bdate_range("2026-01-26", "2026-02-06")
        kept = replay._filter_holdout(dates, include_holdout=True, log=lambda *a: None)
        self.assertEqual(len(kept), len(dates))

    def test_logs_when_holdout_dates_are_skipped(self):
        dates = pd.bdate_range("2026-01-26", "2026-02-06")
        logs = []
        replay._filter_holdout(dates, include_holdout=False, log=logs.append)
        self.assertTrue(any("ホールドアウト" in m for m in logs))


class TruncateBeforeHoldoutTest(unittest.TestCase):
    def test_drops_rows_on_or_after_holdout_start(self):
        idx = pd.bdate_range("2026-01-01", periods=60)
        df = pd.DataFrame({"Close": np.arange(60.0)}, index=idx)
        out = replay._truncate_before_holdout({"1301.T": df})
        self.assertTrue((out["1301.T"].index < replay.HOLDOUT_WINDOW[0]).all())
        self.assertLess(len(out["1301.T"]), len(df))

    def test_none_or_empty_df_passes_through(self):
        out = replay._truncate_before_holdout({"1301.T": None})
        self.assertIsNone(out["1301.T"])


class HoldoutDataLeakTest(unittest.TestCase):
    """日付範囲のフィルタだけでは防げない先読み: ホールドアウト直前のTでも、
    T+hラベルがホールドアウト側のバーを読んではいけない（CLAUDE.md 絶対規則）。"""

    def test_near_boundary_label_does_not_leak_into_holdout(self):
        date_t = pd.Timestamp("2026-01-28")  # HOLDOUT_WINDOW[0]=2026-02-01の直前
        n_bars = 320
        tickers = [f"{1301 + i * 37:04d}.T" for i in range(30)]
        ohlcv_upto_t = make_synthetic(tickers, n_bars=n_bars, seed=0, end=date_t)
        idx_upto_t = make_synthetic_index(n_bars, seed=0, end=date_t)
        listed = _listed(tickers)
        # dateTより先、ホールドアウトの奥まで伸びる「未来」データ（打ち切りが無ければ
        # r_h の計算に使われてしまうはずのデータ）
        ohlcv_full = {t: _extend(ohlcv_upto_t[t], 60, int(t[:4])) for t in tickers}

        # 打ち切り無しで直接計算した場合（先読み保護が無ければ生じる値）を確認しておく
        raw_result = replay.replay_one_day(date_t, ohlcv_full, idx_upto_t, listed,
                                           k=3, label_n=15, pool_days=1, log=lambda *a: None)
        if len(raw_result) == 0:
            self.skipTest("合成データでこの日の採点対象が無かった")
        self.assertTrue(raw_result["r_10"].notna().any(),
                        "テスト設定自体が打ち切り無しでr_10を出せる状況になっていない")

        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv_full, idx_upto_t, listed, tmp, date_t, date_t,
                              k=3, label_n=15, pool_days=1, min_history_bars=250,
                              include_holdout=False, log=lambda *a: None)
            pool = replay.load_replay_table(tmp)

        self.assertGreater(len(pool), 0)
        # run_replay（ホールドアウト保護あり）は、同じ「未来データ付き」ohlcvを渡しても
        # ホールドアウト側を読めないよう内部で打ち切るため、h=10以上はNaNになるはず
        for h in (10, 15, 20, 30):
            self.assertTrue(pool[f"r_{h}"].isna().all(), msg=f"r_{h} leaked past holdout")


def _short_replay_inputs():
    n_bars = 280
    end = pd.Timestamp("2021-08-10")
    tickers = [f"{1301 + i * 37:04d}.T" for i in range(10)]
    ohlcv = make_synthetic(tickers, n_bars=n_bars, seed=1, end=end)
    idx_ohlcv = make_synthetic_index(n_bars, seed=1, end=end)
    listed = _listed(tickers)
    return ohlcv, idx_ohlcv, listed


class ResumeTest(unittest.TestCase):
    """受け入れ: 中断再開できる（既に保存済みの日は再計算しない）。"""

    def test_existing_day_files_are_skipped_on_rerun(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        start, stop = pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-06")
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, start, stop,
                              k=3, label_n=15, pool_days=5, min_history_bars=50, log=lambda *a: None)
            files_before = sorted(Path(tmp).glob("replay_*.csv.gz"))
            self.assertGreater(len(files_before), 0)
            mtimes_before = {f.name: f.stat().st_mtime_ns for f in files_before}

            logs = []
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, start, stop,
                              k=3, label_n=15, pool_days=5, min_history_bars=50, log=logs.append)
            files_after = sorted(Path(tmp).glob("replay_*.csv.gz"))
            mtimes_after = {f.name: f.stat().st_mtime_ns for f in files_after}

            self.assertEqual(mtimes_before, mtimes_after, "再開時に既存日を書き直していないこと")
            self.assertTrue(any("再開スキップ" in m for m in logs))


class LoadReplayTableTest(unittest.TestCase):
    def test_concatenates_all_saved_days(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp,
                              pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-04"),
                              k=3, label_n=15, pool_days=5, min_history_bars=50, log=lambda *a: None)
            table = replay.load_replay_table(tmp)
            self.assertEqual(list(table.columns), replay.REPLAY_COLS)
            n_saved = len(list(Path(tmp).glob("replay_*.csv.gz")))
            self.assertGreaterEqual(n_saved, 1)

    def test_empty_dir_gives_empty_table_with_full_schema(self):
        with TemporaryDirectory() as tmp:
            table = replay.load_replay_table(Path(tmp) / "does_not_exist")
            self.assertEqual(list(table.columns), replay.REPLAY_COLS)
            self.assertEqual(len(table), 0)


if __name__ == "__main__":
    unittest.main()
