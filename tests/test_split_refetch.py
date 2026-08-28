"""新規Splitsイベント検出時の全履歴再取得（T-402）。ネットワーク不要
（fetch_fn を差し替えて検証する）。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot import cli
from stockbot.config import Settings


def _df(n=10, base=1000.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    p = np.full(n, base)
    return pd.DataFrame({"Open": p, "High": p, "Low": p, "Close": p, "Volume": 1e5,
                         "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


class RefetchNewSplitsFullHistoryTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings(
            data_dir=None, universe_seed_csv=None, history_days=400, history_full_days=2600,
            fetch_deadline_sec=5400, fetch_scope="auto", dryrun=False, market_close_hhmm="15:30",
            jpx_listed_url="", jpx_earnings_url="", jpx_delistings_url="",
            min_adv_jpy=2e8, min_price=200.0, min_history_bars=250, adv_window=20,
            max_staleness_days=7, rev_close_tol=0.005, rev_volume_tol=0.05,
            k=3, pullback_max_days=25, pullback_max_depth=0.25, shallow_depth=0.03,
            deep_depth=0.10, pool_days=20, sector_cap=3, corr_threshold=0.8, top_n=10, label_n=15,
        )
        self.now = pd.Timestamp("2026-08-27 16:00", tz="Asia/Tokyo")

    def test_refetches_only_tickers_with_unadjusted_split(self):
        issues = pd.DataFrame([
            {"ticker": "9900.T", "date": pd.Timestamp("2026-08-25"), "kind": "unadjusted_split",
             "ratio": 2.0, "observed": 1.98, "action": "adjusted_prior_rows"},
            {"ticker": "1301.T", "date": pd.Timestamp("2026-08-20"), "kind": "suspected_unrecorded_split",
             "ratio": 3.0, "observed": 2.95, "action": "flag_only"},
        ])
        ohlcv = {"9900.T": _df(base=900.0), "1301.T": _df(base=1500.0)}

        calls = []

        def fake_fetch(tickers, history_days, deadline_sec, now_jst=None, close_hhmm=None, log=print):
            calls.append((list(tickers), history_days))
            full = _df(n=2600, base=850.0)
            return {"9900.T": full}, {"data_total": 1, "data_ok": 1}

        out_ohlcv, out_issues = cli._refetch_new_splits_full_history(
            ohlcv, issues, self.cfg, self.now, log=lambda *a: None, fetch_fn=fake_fetch)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (["9900.T"], 2600))  # 1301.Tは呼ばれない
        self.assertEqual(len(out_ohlcv["9900.T"]), 2600)  # 全履歴で置き換わっている
        self.assertEqual(len(out_ohlcv["1301.T"]), 10)  # 対象外は元のまま
        # 9900.Tのunadjusted_split行は消え、1301.Tのsuspected行はそのまま残る
        self.assertNotIn("9900.T", set(out_issues.loc[out_issues["kind"] == "unadjusted_split", "ticker"]))
        self.assertIn("1301.T", set(out_issues["ticker"]))

    def test_no_op_when_no_unadjusted_split_issues(self):
        issues = pd.DataFrame([
            {"ticker": "1301.T", "date": pd.Timestamp("2026-08-20"), "kind": "suspected_unrecorded_split",
             "ratio": 3.0, "observed": 2.95, "action": "flag_only"},
        ])
        ohlcv = {"1301.T": _df()}
        calls = []

        def fake_fetch(*a, **kw):
            calls.append(True)
            return {}, {}

        out_ohlcv, out_issues = cli._refetch_new_splits_full_history(
            ohlcv, issues, self.cfg, self.now, log=lambda *a: None, fetch_fn=fake_fetch)
        self.assertEqual(len(calls), 0)
        self.assertIs(out_ohlcv, ohlcv)
        self.assertTrue(out_issues.equals(issues))

    def test_no_op_when_issues_empty(self):
        issues = pd.DataFrame(columns=["ticker", "date", "kind", "ratio", "observed", "action"])
        ohlcv = {"1301.T": _df()}
        calls = []

        def fake_fetch(*a, **kw):
            calls.append(True)
            return {}, {}

        out_ohlcv, out_issues = cli._refetch_new_splits_full_history(
            ohlcv, issues, self.cfg, self.now, log=lambda *a: None, fetch_fn=fake_fetch)
        self.assertEqual(len(calls), 0)
        self.assertIs(out_ohlcv, ohlcv)


if __name__ == "__main__":
    unittest.main()
