"""結果付け（docs/SCREENER.md §3.3）。

窓の固定（あとから足が伸びても確定済みの結果が動かない）と、打ち切りの扱いを
必須テストとして含む。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.store import IDX_TICKER
from stockbot.screener.record import DELIVERED_COLS, records_to_frame, save_delivered
from stockbot.screener.resolver import (
    HORIZON_DAYS,
    due_date,
    load_journal,
    load_outcome,
    outcome_path,
    resolve_delivered,
    resolve_pending,
    resolve_row,
    save_outcome,
    summarize_by_landing_ma,
    trading_calendar,
)

WARMUP = 20            # 平坦な助走（SMA5 = 100）
ASOF_POS = WARMUP - 1  # 判定日 T
DATES = pd.bdate_range("2024-01-01", periods=40)
ASOF = DATES[ASOF_POS]
DELIVERED_ON = DATES[ASOF_POS + 1]


def make_frame(after_closes, n_after=None):
    """助走 20 本（終値 100 で平坦）＋ T+1 以降の終値。High=C+1 / Low=C−1 / Open=前日終値。"""
    closes = np.concatenate([np.full(WARMUP, 100.0), np.asarray(after_closes, dtype=float)])
    if n_after is not None:
        closes = closes[: WARMUP + n_after]
    idx = DATES[: len(closes)]
    close = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(100.0),
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
    }, index=idx)


def make_record(ticker="1234.T", lp=95.0, h0_high=106.0, landing_ma="SMA25"):
    return {
        "delivered_on": DELIVERED_ON, "asof": ASOF, "ticker": ticker, "name": "テスト",
        "landing_ma": landing_ma, "landing_ma_value": 99.0, "landing_dist_atr": 0.5,
        "lp": lp, "lp_date": DATES[ASOF_POS - 2],
        "h0_high": h0_high, "h0_date": DATES[ASOF_POS - 8],
        "close_t": 100.0, "atr_t": 2.0, "state": "形成中",
        "depth_pct": 0.05, "pullback_days": 5,
    }


def calendar_ohlcv(frames):
    """銘柄フレーム群に、営業日軸用の指数（40 本）を足した store 相当の dict。"""
    idx_df = make_frame(np.full(20, 100.0))
    out = {IDX_TICKER: idx_df}
    out.update(frames)
    return out


class TestResolveRow(unittest.TestCase):
    def test_success_when_high_updated_without_breaking_lp(self):
        df = make_frame([101, 103, 105, 107, 109])
        out = resolve_row(pd.Series(make_record()), df)
        self.assertEqual(out["n_bars"], HORIZON_DAYS)
        self.assertFalse(out["censored"])
        self.assertAlmostEqual(out["entry_open"], 100.0)
        self.assertAlmostEqual(out["close_h"], 109.0)
        self.assertAlmostEqual(out["max_high"], 110.0)
        self.assertAlmostEqual(out["min_low"], 100.0)
        self.assertAlmostEqual(out["ret_h"], 0.09)
        self.assertFalse(out["broke_lp"])
        self.assertTrue(out["reached_h0"])
        self.assertEqual(out["reached_h0_day"], 4)
        self.assertTrue(out["recovered_sma5"])
        self.assertEqual(out["recovered_sma5_day"], 1)
        self.assertTrue(out["success"])

    def test_failure_when_lp_broken_and_high_not_updated(self):
        df = make_frame([98, 94, 96, 97, 99])
        out = resolve_row(pd.Series(make_record()), df)
        self.assertTrue(out["broke_lp"])
        self.assertEqual(out["broke_lp_day"], 2)
        self.assertFalse(out["reached_h0"])
        self.assertFalse(out["success"])

    def test_failure_when_lp_broken_before_high_updated(self):
        df = make_frame([94, 96, 100, 105, 110])
        out = resolve_row(pd.Series(make_record()), df)
        self.assertEqual(out["broke_lp_day"], 1)
        self.assertEqual(out["reached_h0_day"], 5)
        self.assertFalse(out["success"])

    def test_sma5_not_recovered_while_falling(self):
        df = make_frame([99, 98, 97, 96, 95])
        out = resolve_row(pd.Series(make_record()), df)
        self.assertFalse(out["recovered_sma5"])
        self.assertTrue(pd.isna(out["recovered_sma5_day"]))

    def test_censored_when_fewer_bars(self):
        df = make_frame([101, 103, 105], n_after=3)
        out = resolve_row(pd.Series(make_record()), df)
        self.assertEqual(out["n_bars"], 3)
        self.assertTrue(out["censored"])
        self.assertAlmostEqual(out["close_h"], 105.0)   # 取れた最後の終値で確定させる

    def test_missing_ticker_is_censored_not_an_error(self):
        out = resolve_row(pd.Series(make_record()), None)
        self.assertEqual(out["n_bars"], 0)
        self.assertTrue(out["censored"])
        self.assertTrue(np.isnan(out["entry_open"]))
        self.assertTrue(pd.isna(out["success"]))

    def test_missing_lp_and_h0_leave_those_judgements_missing(self):
        df = make_frame([101, 103, 105, 107, 109])
        row = pd.Series(dict(make_record(), lp=np.nan, h0_high=np.nan))
        out = resolve_row(row, df)
        self.assertEqual(out["n_bars"], HORIZON_DAYS)
        self.assertTrue(pd.isna(out["broke_lp"]))
        self.assertTrue(pd.isna(out["reached_h0"]))
        self.assertTrue(pd.isna(out["success"]))
        self.assertTrue(out["recovered_sma5"])   # 5日線の判定は記録に依存しないので出る

    def test_asof_missing_from_series_is_censored(self):
        df = make_frame([101, 103, 105, 107, 109]).drop(index=ASOF)
        out = resolve_row(pd.Series(make_record()), df)
        self.assertEqual(out["n_bars"], 0)
        self.assertTrue(out["censored"])

    def test_window_is_fixed_at_horizon(self):
        """あとから足が伸びても、確定した結果は変わらない（再計算一致、CLAUDE.md）。"""
        short = make_frame([101, 103, 105, 107, 109])
        long = make_frame([101, 103, 105, 107, 109, 130, 60, 130, 60, 130])
        a = resolve_row(pd.Series(make_record()), short)
        b = resolve_row(pd.Series(make_record()), long)
        for key in ("n_bars", "censored", "entry_open", "close_h", "max_high", "min_low",
                    "ret_h", "broke_lp", "reached_h0_day", "recovered_sma5_day", "success"):
            self.assertEqual(str(a[key]), str(b[key]), key)


class TestCalendarAndPending(unittest.TestCase):
    def test_trading_calendar_prefers_index(self):
        ohlcv = calendar_ohlcv({"1234.T": make_frame([101], n_after=1)})
        cal = trading_calendar(ohlcv)
        self.assertEqual(len(cal), 40)
        self.assertEqual(cal[0], DATES[0])

    def test_trading_calendar_falls_back_to_union(self):
        cal = trading_calendar({"1234.T": make_frame([101, 102], n_after=2)})
        self.assertEqual(len(cal), WARMUP + 2)

    def test_due_date_counts_market_business_days(self):
        cal = trading_calendar(calendar_ohlcv({}))
        self.assertEqual(due_date(cal, ASOF, 5), DATES[ASOF_POS + 5])
        self.assertIsNone(due_date(cal, DATES[38], 5))       # 5 本先がまだ無い
        self.assertIsNone(due_date(cal, pd.Timestamp("2024-01-06"), 5))  # 軸に無い日

    def _write_delivered(self, daily, ticker="1234.T", landing_ma="SMA25"):
        df = records_to_frame([make_record(ticker=ticker, landing_ma=landing_ma)])
        self.assertEqual(list(df.columns), DELIVERED_COLS)
        return save_delivered(df, daily, DELIVERED_ON)

    def test_resolve_pending_writes_outcome_when_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            self._write_delivered(daily)
            ohlcv = calendar_ohlcv({"1234.T": make_frame([101, 103, 105, 107, 109])})
            written = resolve_pending(daily, ohlcv, log=lambda *_a: None)
            self.assertEqual([p.name for p in written],
                             [f"outcome_{DELIVERED_ON.strftime('%Y-%m-%d')}.csv"])
            out = load_outcome(written[0])
            self.assertEqual(len(out), 1)
            self.assertTrue(bool(out["success"].iloc[0]))
            self.assertEqual(out["resolved_on"].iloc[0], DATES[ASOF_POS + 5])

    def test_resolve_pending_skips_when_not_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            self._write_delivered(daily)
            # 指数を T+3 までしか持たない = 5 営業日がまだ経過していない
            ohlcv = {IDX_TICKER: make_frame(np.full(3, 100.0), n_after=3),
                     "1234.T": make_frame([101, 103, 105], n_after=3)}
            self.assertEqual(resolve_pending(daily, ohlcv, log=lambda *_a: None), [])
            self.assertFalse(outcome_path(daily, DELIVERED_ON).exists())

    def test_resolve_pending_does_not_redo_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            self._write_delivered(daily)
            ohlcv = calendar_ohlcv({"1234.T": make_frame([101, 103, 105, 107, 109])})
            resolve_pending(daily, ohlcv, log=lambda *_a: None)
            before = outcome_path(daily, DELIVERED_ON).read_text()
            # 値が動く別データで再実行しても、確定済みのファイルは作り直さない
            ohlcv2 = calendar_ohlcv({"1234.T": make_frame([90, 80, 70, 60, 50])})
            self.assertEqual(resolve_pending(daily, ohlcv2, log=lambda *_a: None), [])
            self.assertEqual(outcome_path(daily, DELIVERED_ON).read_text(), before)

    def test_resolve_pending_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_pending(Path(tmp), {}, log=lambda *_a: None), [])


class TestJournal(unittest.TestCase):
    def _build(self, daily):
        records = [make_record(ticker="1111.T", landing_ma="SMA25"),
                   make_record(ticker="2222.T", landing_ma="SMA75")]
        save_delivered(records_to_frame(records), daily, DELIVERED_ON)
        ohlcv = calendar_ohlcv({
            "1111.T": make_frame([101, 103, 105, 107, 109]),   # 成功
            "2222.T": make_frame([98, 94, 96, 97, 99]),        # 押し安値割れ
        })
        resolve_pending(daily, ohlcv, log=lambda *_a: None)

    def test_journal_joins_delivered_and_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            self._build(daily)
            j = load_journal(daily)
            self.assertEqual(len(j), 2)
            self.assertEqual(set(j["ticker"]), {"1111.T", "2222.T"})
            row = j[j["ticker"] == "1111.T"].iloc[0]
            self.assertEqual(row["landing_ma"], "SMA25")
            self.assertTrue(bool(row["success"]))

    def test_journal_keeps_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            save_delivered(records_to_frame([make_record()]), daily, DELIVERED_ON)
            j = load_journal(daily)
            self.assertEqual(len(j), 1)
            self.assertNotIn("success", j.columns)

    def test_summary_by_landing_ma(self):
        """4本を常に出す。候補に出ていない線は n=0（docs/SCREENER.md §3.4）。"""
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            self._build(daily)
            s = summarize_by_landing_ma(load_journal(daily))
            self.assertEqual(s["landing_ma"].tolist(), ["SMA5", "SMA25", "SMA75", "SMA200"])
            self.assertEqual(s["n"].tolist(), [0, 1, 1, 0])
            self.assertEqual(s["n_resolved"].tolist(), [0, 1, 1, 0])
            by = s.set_index("landing_ma")
            self.assertAlmostEqual(by.loc["SMA25", "success_rate"], 1.0)
            self.assertAlmostEqual(by.loc["SMA75", "success_rate"], 0.0)
            self.assertAlmostEqual(by.loc["SMA75", "broke_lp_rate"], 1.0)
            # 候補に出ていない線は率が欠損（0.0 ではない。0勝ではなく未出走）
            for ma in ("SMA5", "SMA200"):
                self.assertTrue(pd.isna(by.loc[ma, "success_rate"]), ma)
                self.assertTrue(pd.isna(by.loc[ma, "mean_ret_h"]), ma)

    def test_summary_filters_by_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            far = dict(make_record(ticker="3333.T", landing_ma="SMA200"), landing_dist_atr=9.0)
            save_delivered(records_to_frame([make_record(ticker="1111.T"), far]),
                           daily, DELIVERED_ON)
            j = load_journal(daily)
            self.assertEqual(summarize_by_landing_ma(j)["n"].tolist(), [0, 1, 0, 1])
            # 距離で絞ると SMA200 の1件が落ちるが、行は n=0 で残る
            filtered = summarize_by_landing_ma(j, max_dist_atr=1.0)
            self.assertEqual(filtered["landing_ma"].tolist(),
                             ["SMA5", "SMA25", "SMA75", "SMA200"])
            self.assertEqual(filtered["n"].tolist(), [0, 1, 0, 0])

    def test_summary_on_empty_journal(self):
        """記録が無い日も 4 行の形は保つ（全て n=0）。ただしこれは D4 の情報ではない。"""
        s = summarize_by_landing_ma(pd.DataFrame())
        self.assertEqual(s["landing_ma"].tolist(), ["SMA5", "SMA25", "SMA75", "SMA200"])
        self.assertEqual(s["n"].tolist(), [0, 0, 0, 0])
        self.assertIn("success_rate", s.columns)
        self.assertTrue(s["success_rate"].isna().all())

    def test_summary_keeps_no_line_rows_when_present(self):
        """止まった線が取れなかった記録がある日だけ、末尾に「（線なし）」を足す。"""
        from stockbot.screener.resolver import NO_LINE

        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            recs = [make_record(ticker="1111.T", landing_ma="SMA25"),
                    dict(make_record(ticker="4444.T"), landing_ma="")]
            save_delivered(records_to_frame(recs), daily, DELIVERED_ON)
            s = summarize_by_landing_ma(load_journal(daily))
            self.assertEqual(s["landing_ma"].tolist(),
                             ["SMA5", "SMA25", "SMA75", "SMA200", NO_LINE])
            self.assertEqual(int(s.set_index("landing_ma").loc[NO_LINE, "n"]), 1)

    def test_outcome_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            delivered = records_to_frame([make_record()])
            ohlcv = {"1234.T": make_frame([101, 103, 105, 107, 109])}
            out = resolve_delivered(delivered, ohlcv, resolved_on=DATES[ASOF_POS + 5])
            path = save_outcome(out, daily, DELIVERED_ON)
            back = load_outcome(path)
            self.assertEqual(back["ticker"].iloc[0], "1234.T")
            self.assertTrue(bool(back["success"].iloc[0]))
            self.assertFalse(bool(back["censored"].iloc[0]))
            self.assertAlmostEqual(float(back["ret_h"].iloc[0]), 0.09)


if __name__ == "__main__":
    unittest.main()
