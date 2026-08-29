"""歴史的再生（DESIGN.md §10.1・§10.5 / TASKS.md T-402）のテスト。"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

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


class RealTradingDaysTest(unittest.TestCase):
    """T-402: pd.bdate_range は祝日を除外しないため、休場日でも少数銘柄に
    データの乱れがあると幻の候補日が生じる（2018-07-16=海の日で確認済み）。"""

    def test_real_trading_day_kept_phantom_day_dropped(self):
        dates = pd.bdate_range("2018-07-13", "2018-07-18")
        real_day = pd.Timestamp("2018-07-13")
        phantom_day = pd.Timestamp("2018-07-16")  # 海の日（月曜、休場）
        ohlcv = {}
        for i in range(10):
            idx = dates.drop(phantom_day) if i >= 2 else dates  # 2/10銘柄だけ幻日にもデータを持つ
            ohlcv[f"T{i}"] = pd.DataFrame({"Close": 1.0}, index=idx)
        kept = replay._real_trading_days(dates, ohlcv, min_frac=0.5, log=lambda *a: None)
        self.assertIn(real_day, kept)
        self.assertNotIn(phantom_day, kept)

    def test_logs_dropped_dates(self):
        dates = pd.bdate_range("2018-07-13", "2018-07-17")
        phantom_day = pd.Timestamp("2018-07-16")
        ohlcv = {f"T{i}": pd.DataFrame({"Close": 1.0}, index=dates.drop(phantom_day)) for i in range(10)}
        logs = []
        replay._real_trading_days(dates, ohlcv, min_frac=0.5, log=logs.append)
        self.assertTrue(any("2018-07-16" in m for m in logs))

    def test_empty_ohlcv_returns_dates_unchanged(self):
        dates = pd.bdate_range("2018-07-13", "2018-07-17")
        kept = replay._real_trading_days(dates, {}, log=lambda *a: None)
        self.assertTrue((kept == dates).all())

    def test_threshold_boundary_is_inclusive(self):
        dates = pd.DatetimeIndex([pd.Timestamp("2018-07-13")])
        ohlcv = {"T0": pd.DataFrame({"Close": 1.0}, index=dates),
                "T1": pd.DataFrame({"Close": 1.0}, index=pd.DatetimeIndex([]))}
        kept = replay._real_trading_days(dates, ohlcv, min_frac=0.5, log=lambda *a: None)
        self.assertEqual(len(kept), 1)  # frac=0.5、>= 0.5 は保持


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


class ChunkedReplayPoolContinuityTest(unittest.TestCase):
    """DESIGN.md §6.1: 直近20営業日プールが、run_replay を期間で区切って複数回に
    分けて実行しても途切れないこと（年ごとの分割実行を想定した確認依頼への対応）。

    「2021-08〜2026-02 を一括で実行した結果」と「年ごとに分割実行して結合した結果」の
    score_v1 が一致することを、3ヶ月を2分割した短い期間で確認する。
    """

    def test_split_execution_matches_single_run(self):
        # min_history_bars=250 かつ window_start が window_end の約42営業日前なので、
        # n_bars はそれより十分大きくないと window_start 付近でユニバースが空になり
        # （経過履歴不足）、EmptyDayRateGuardTest の閾値に引っかかって窓全体の検証に
        # ならない。300以上の余裕を持たせる
        n_bars = 320
        window_start = pd.Timestamp("2022-01-03")
        split = pd.Timestamp("2022-01-31")  # 区切りの境界（この日で前半が終わる）
        window_end = pd.Timestamp("2022-03-01")
        tickers = [f"{1301 + i * 37:04d}.T" for i in range(20)]

        ohlcv_upto = make_synthetic(tickers, n_bars=n_bars, seed=0, end=window_end)
        idx_ohlcv = make_synthetic_index(n_bars, seed=0, end=window_end)
        listed = _listed(tickers)
        ohlcv_full = {t: _extend(ohlcv_upto[t], 30, int(t[:4])) for t in tickers}

        common_kwargs = dict(k=3, label_n=15, pool_days=10, min_history_bars=250,
                             log=lambda *a: None)

        with TemporaryDirectory() as tmp_combined, TemporaryDirectory() as tmp_split:
            # 一括実行
            replay.run_replay(ohlcv_full, idx_ohlcv, listed, tmp_combined,
                              window_start, window_end, **common_kwargs)
            pool_combined = replay.load_replay_table(tmp_combined)

            # 分割実行（同じ output_dir に前半・後半を続けて書く。CIで年ごとに
            # トリガーする運用を模している）
            replay.run_replay(ohlcv_full, idx_ohlcv, listed, tmp_split,
                              window_start, split, **common_kwargs)
            next_day = split + pd.tseries.offsets.BDay(1)
            replay.run_replay(ohlcv_full, idx_ohlcv, listed, tmp_split,
                              next_day, window_end, **common_kwargs)
            pool_split = replay.load_replay_table(tmp_split)

        self.assertGreater(len(pool_combined), 0)
        a = pool_combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        b = pool_split.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.assertEqual(len(a), len(b))
        self.assertEqual(list(a["ticker"]), list(b["ticker"]))
        self.assertEqual(list(a["date"]), list(b["date"]))

        # CSVへの保存・再読み込みを挟む分、float64のビット単位の完全一致にはならない
        # （pandas の to_csv/read_csv は既定で丸め誤差を持つ。T-301 の
        # load_recent_daily_features も同じ制約を持つ既存の性質で、このテストが
        # 検出したいのは「プールが境界で丸ごと空になる」ような大きな不一致）
        diff = (a["score_v1"] - b["score_v1"]).abs()
        self.assertLess(diff.max(), 1.0, msg="score_v1 が一括実行と分割実行で大きく食い違う")
        # 境界直後の日だけを見ても同様（ここがプール断絶の影響を最も受けやすい）
        near_boundary = a[pd.to_datetime(a["date"]).between(split, split + pd.Timedelta(days=10))]
        if len(near_boundary):
            idx = near_boundary.index
            boundary_diff = (a.loc[idx, "score_v1"] - b.loc[idx, "score_v1"]).abs()
            self.assertLess(boundary_diff.max(), 1.0,
                            msg="区切り直後のscore_v1が一括実行と食い違う（プール断絶の疑い）")


def _short_replay_inputs():
    n_bars = 280
    end = pd.Timestamp("2021-08-10")
    tickers = [f"{1301 + i * 37:04d}.T" for i in range(10)]
    ohlcv = make_synthetic(tickers, n_bars=n_bars, seed=1, end=end)
    idx_ohlcv = make_synthetic_index(n_bars, seed=1, end=end)
    listed = _listed(tickers)
    return ohlcv, idx_ohlcv, listed


def _crash_tail(df: pd.DataFrame, n_crash: int) -> pd.DataFrame:
    """末尾 n_crash 本を急落させ、Close が SMA200 を大きく割り込むようにする
    （G2: Close>=SMA200 が全銘柄で落ちる地合い急変を模したテスト用ヘルパー）。
    O/H/L/Close を同じ係数で一律にスケールするので、日中の高安関係は保たれる。"""
    df = df.copy()
    factor = np.exp(np.linspace(-0.1, -0.9, n_crash))
    for col in ("Open", "High", "Low", "Close"):
        loc = df.columns.get_loc(col)
        df.iloc[-n_crash:, loc] = df.iloc[-n_crash:][col].to_numpy() * factor
    return df


def _crashed_replay_inputs(n_crash=15):
    ohlcv, idx_ohlcv, listed = _short_replay_inputs()
    crashed = {t: _crash_tail(df, n_crash) for t, df in ohlcv.items()}
    return crashed, idx_ohlcv, listed


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


class EmptyDayRateGuardTest(unittest.TestCase):
    """2026-08-24 の事故（指数の長期履歴が無く全営業日が空のまま run_replay が成功
    終了していた）の再発防止。空(採点対象0件)の日が MAX_EMPTY_DAY_RATE を超えたら
    run_replay は RuntimeError を送出しなければならない。"""

    def test_normal_run_does_not_raise(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        start, stop = pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-10")
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, start, stop,
                              k=3, label_n=15, pool_days=5, min_history_bars=50, log=lambda *a: None)
            table = replay.load_replay_table(tmp)
            self.assertGreater(len(table), 0, "合成データなら空の日ばかりにはならないはず")

    def test_mostly_missing_index_data_raises(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        # 指数データを先頭数本だけに切り詰める(= backfillでindexの長期履歴を取り忘れた事故を再現)
        idx_short = idx_ohlcv.iloc[:3]
        start, stop = pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-10")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                replay.run_replay(ohlcv, idx_short, listed, tmp, start, stop,
                                  k=3, label_n=15, pool_days=5, min_history_bars=50,
                                  log=lambda *a: None)
            self.assertIn("異常終了", str(ctx.exception))


class EmptyReasonClassificationTest(unittest.TestCase):
    """空(0行)の日には「データが無い」（EMPTY_REASON_NO_DATA）と「ゲート等で
    候補が0件」（EMPTY_REASON_NO_CANDIDATES）の2種類があり、MAX_EMPTY_DAY_RATE
    ガードは前者だけで判定しなければならない。2020-02〜06 のような地合い急落時に
    G2（Close>=SMA200）で全銘柄が落ちるのは正しい挙動であり、データの不備ではない
    （2026-08-26 の実行条件確認への対応）。"""

    def test_all_gate_failures_do_not_raise_even_at_100_percent(self):
        ohlcv, idx_ohlcv, listed = _crashed_replay_inputs(n_crash=15)
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=8)
        with TemporaryDirectory() as tmp:
            # 全日が「候補0件」でも例外を投げないことそのものが検証内容
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=lambda *a: None)
            table = replay.load_replay_table(tmp)
            self.assertEqual(len(table), 0, "全銘柄がG2で落ちるはずなので採点対象は0件")

    def test_meta_sidecar_marks_gate_failures_as_no_candidates(self):
        ohlcv, idx_ohlcv, listed = _crashed_replay_inputs(n_crash=15)
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=8)
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=lambda *a: None)
            metas = list(Path(tmp).glob("replay_meta_*.json"))
            self.assertGreater(len(metas), 0, "候補0件の日のサイドカーが書かれているはず")
            for p in metas:
                self.assertEqual(json.loads(p.read_text())["empty_reason"],
                                 replay.EMPTY_REASON_NO_CANDIDATES)

    def test_meta_sidecar_survives_resume_without_raising(self):
        """中断再開（同じ範囲を再実行）しても「候補0件」の分類が保たれ、
        ガードの判定が変わらないこと（.attrs はCSVに残らないため、サイドカーを
        正しく読み戻せているかがここで初めて試される）。"""
        ohlcv, idx_ohlcv, listed = _crashed_replay_inputs(n_crash=15)
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=8)
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=lambda *a: None)
            logs = []
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=logs.append)
            self.assertTrue(any("再開スキップ" in m for m in logs))

    def test_a_minority_of_true_no_data_days_mixed_with_gate_failures_does_not_raise(self):
        """8日中1日だけ指数データが欠落（12.5%、閾値0.5未満）、残り7日はゲート落ちに
        よる空日。指数欠損の比率だけで判定すれば例外にならないはず。"""
        ohlcv, idx_ohlcv, listed = _crashed_replay_inputs(n_crash=15)
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=8)
        idx_gapped = idx_ohlcv.drop(index=[dates[0]])
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_gapped, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=lambda *a: None)

    def test_a_majority_of_true_no_data_days_still_raises_when_mixed(self):
        """8日中5日（62.5%、閾値0.5超）で指数データが欠落していれば、残りがゲート落ち
        による空日であっても、指数欠損の比率だけで正しくガードが発火すること。"""
        ohlcv, idx_ohlcv, listed = _crashed_replay_inputs(n_crash=15)
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=8)
        idx_gapped = idx_ohlcv.drop(index=list(dates[:5]))
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                replay.run_replay(ohlcv, idx_gapped, listed, tmp, dates[0], dates[-1],
                                  k=3, label_n=15, pool_days=5, min_history_bars=50,
                                  log=lambda *a: None)
            self.assertIn("異常終了", str(ctx.exception))


class WarmupTaggingTest(unittest.TestCase):
    """DESIGN.md §10.1「開始日の規則（R1.1で追加）」: stats_start より前
    （G3の252本ルックバック＋§6.3プールの20日ウォームアップの期間）は
    warmup=True で出力し、集計から除外できるようにする。"""

    def test_rows_before_stats_start_are_flagged(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        dates = pd.bdate_range(end=pd.Timestamp("2021-08-10"), periods=6)
        stats_start = dates[3]
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp, dates[0], dates[-1],
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              stats_start=stats_start, log=lambda *a: None)
            table = replay.load_replay_table(tmp)
            self.assertIn("warmup", table.columns)
            table_dates = pd.to_datetime(table["date"])
            before = table[table_dates < stats_start]
            on_or_after = table[table_dates >= stats_start]
            self.assertGreater(len(before) + len(on_or_after), 0)
            if len(before):
                self.assertTrue(bool(before["warmup"].all()))
            if len(on_or_after):
                self.assertFalse(bool(on_or_after["warmup"].any()))

    def test_no_stats_start_means_no_warmup_column(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp,
                              pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-04"),
                              k=3, label_n=15, pool_days=5, min_history_bars=50,
                              log=lambda *a: None)
            table = replay.load_replay_table(tmp)
            self.assertNotIn("warmup", table.columns)


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


class LoadRecentReplayDaysTest(unittest.TestCase):
    """区切り実行の境界でプールが途切れないようにする _load_recent_replay_days。"""

    def test_loads_only_days_strictly_before_cutoff(self):
        ohlcv, idx_ohlcv, listed = _short_replay_inputs()
        with TemporaryDirectory() as tmp:
            replay.run_replay(ohlcv, idx_ohlcv, listed, tmp,
                              pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-10"),
                              k=3, label_n=15, pool_days=5, min_history_bars=50, log=lambda *a: None)
            saved_dates = sorted(f.name for f in Path(tmp).glob("replay_*.csv.gz"))
            self.assertGreater(len(saved_dates), 3)

            recent = replay._load_recent_replay_days(tmp, pd.Timestamp("2021-08-10"), pool_days=3)
            self.assertEqual(len(recent), 2)  # pool_days-1
            for df in recent:
                self.assertEqual(list(df.columns), replay.DAILY_FEATURES_COLS)
                self.assertTrue((pd.to_datetime(df["date"]) < pd.Timestamp("2021-08-10")).all())

    def test_pool_days_one_returns_empty(self):
        recent = replay._load_recent_replay_days("/does/not/matter", pd.Timestamp("2021-08-10"),
                                                  pool_days=1)
        self.assertEqual(recent, [])

    def test_missing_dir_returns_empty(self):
        recent = replay._load_recent_replay_days("/does/not/exist/at/all",
                                                  pd.Timestamp("2021-08-10"), pool_days=5)
        self.assertEqual(recent, [])


class APrimePopulationTest(unittest.TestCase):
    """DESIGN.md §10.2-4 (a′): G0〜G3通過だが押し目状態でない銘柄だけを選ぶこと
    （T-403、§9.3のhの選択に使う母集団）。pullback_state/evaluate_gates を差し替えて
    母集団の反転ロジックだけを検証する（実データで特定のstate/gate_passを作るのは
    困難なため）。"""

    def test_included_when_gate_pass_true_and_state_not_scorable(self):
        ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        date_t = pd.Timestamp("2021-08-10")
        with mock.patch.object(replay.pullback, "pullback_state",
                               return_value={"state": replay.pullback.STATE_NO_STRUCTURE}), \
             mock.patch.object(replay.gates, "evaluate_gates", return_value={"gate_pass": True}):
            out = replay._a_prime_one_day(date_t, ohlcv, listed, k=3, label_n=15, min_history_bars=50)
        self.assertEqual(sorted(out["ticker"]), sorted(ohlcv))
        self.assertEqual(list(out.columns), replay.A_PRIME_COLS)

    def test_excluded_when_state_is_scorable_even_if_gate_pass_true(self):
        # 状態が採点対象（形成中/反発開始/ブレイク）なら、ゲート判定を待たず除外される
        # （評価対象は「押し目状態でない」銘柄だけ、DESIGN.md §10.2-4(a′)）
        ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        date_t = pd.Timestamp("2021-08-10")
        with mock.patch.object(replay.pullback, "pullback_state",
                               return_value={"state": replay.pullback.STATE_FORMING}), \
             mock.patch.object(replay.gates, "evaluate_gates", return_value={"gate_pass": True}) as gate_mock:
            out = replay._a_prime_one_day(date_t, ohlcv, listed, k=3, label_n=15, min_history_bars=50)
        self.assertEqual(len(out), 0)
        gate_mock.assert_not_called()

    def test_excluded_when_gate_pass_false(self):
        ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        date_t = pd.Timestamp("2021-08-10")
        with mock.patch.object(replay.pullback, "pullback_state",
                               return_value={"state": replay.pullback.STATE_NO_STRUCTURE}), \
             mock.patch.object(replay.gates, "evaluate_gates", return_value={"gate_pass": False}):
            out = replay._a_prime_one_day(date_t, ohlcv, listed, k=3, label_n=15, min_history_bars=50)
        self.assertEqual(len(out), 0)

    def test_empty_universe_returns_empty_frame_with_schema(self):
        _ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        out = replay._a_prime_one_day(pd.Timestamp("2021-08-10"), {}, listed,
                                      k=3, label_n=15, min_history_bars=50)
        self.assertEqual(list(out.columns), replay.A_PRIME_COLS)
        self.assertEqual(len(out), 0)


class RunAPrimeReplayTest(unittest.TestCase):
    """run_a_prime_replay: 日付ごとの保存・中断再開・load_a_prime_table を確認する。"""

    def test_saves_one_file_per_business_day_and_is_resumable(self):
        ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        start, stop = pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-06")
        with TemporaryDirectory() as tmp:
            replay.run_a_prime_replay(ohlcv, listed, tmp, start, stop,
                                      k=3, label_n=15, min_history_bars=50, log=lambda *a: None)
            files_before = sorted(Path(tmp).glob("a_prime_*.csv.gz"))
            self.assertGreater(len(files_before), 0)
            mtimes_before = {f.name: f.stat().st_mtime_ns for f in files_before}

            logs = []
            replay.run_a_prime_replay(ohlcv, listed, tmp, start, stop,
                                      k=3, label_n=15, min_history_bars=50, log=logs.append)
            files_after = sorted(Path(tmp).glob("a_prime_*.csv.gz"))
            mtimes_after = {f.name: f.stat().st_mtime_ns for f in files_after}

            self.assertEqual(mtimes_before, mtimes_after, "再開時に既存日を書き直していないこと")
            self.assertTrue(any("再開スキップ" in m for m in logs))

    def test_load_a_prime_table_concatenates_all_saved_days(self):
        ohlcv, _idx_ohlcv, listed = _short_replay_inputs()
        with TemporaryDirectory() as tmp:
            replay.run_a_prime_replay(ohlcv, listed, tmp,
                                      pd.Timestamp("2021-08-02"), pd.Timestamp("2021-08-04"),
                                      k=3, label_n=15, min_history_bars=50, log=lambda *a: None)
            table = replay.load_a_prime_table(tmp)
            self.assertEqual(list(table.columns), replay.A_PRIME_COLS)

    def test_load_a_prime_table_empty_dir_gives_empty_schema(self):
        with TemporaryDirectory() as tmp:
            table = replay.load_a_prime_table(Path(tmp) / "does_not_exist")
            self.assertEqual(list(table.columns), replay.A_PRIME_COLS)
            self.assertEqual(len(table), 0)


if __name__ == "__main__":
    unittest.main()
