"""継ぎ目の検査（2026-09-17・1909.T の破損を受けて追加）。

固定するのは 5 つ。**壊れた新規行を store に入れないこと**、**継ぎ目が無い銘柄は
検査しないこと**（新規上場と全履歴再取得の修復経路を止めない）、**分割は素通り
させること**、**出来高ゼロの検査**、**記録を積むこと**。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.store import LONG_COLS, SEAM_RATIO_MAX, OhlcvStore, seam_issues


def long_frame(ticker, start, n, close=1000.0, volume=100000.0, step=0.0):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "open": close, "high": close, "low": close,
        "close": [close + step * i for i in range(n)],
        "volume": volume, "dividends": 0.0, "splits": 0.0})[LONG_COLS]


class SeamIssuesTest(unittest.TestCase):
    """`seam_issues` 単体。**継ぎ目が無ければ何も言わない。**"""

    def setUp(self):
        self.old = long_frame("1909.T", "2024-01-01", 60, close=1000.0)

    def test_healthy_refresh_is_clean(self):
        new = long_frame("1909.T", "2024-03-01", 30, close=1050.0)
        self.assertEqual(len(seam_issues(self.old, new)), 0)

    def test_market_cap_in_the_price_field_is_caught(self):
        """**1909.T の形。** 終値が一定倍率で跳ね上がる。"""
        new = long_frame("1909.T", "2024-03-01", 30, close=1000.0 * 4_399_472)
        bad = seam_issues(self.old, new)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad.iloc[0]["kind"], "seam_close_ratio")
        self.assertGreater(bad.iloc[0]["close_ratio"], SEAM_RATIO_MAX)

    def test_collapse_is_caught_too(self):
        new = long_frame("1909.T", "2024-03-01", 30, close=1000.0 / 500)
        self.assertEqual(seam_issues(self.old, new).iloc[0]["kind"], "seam_close_ratio")

    def test_a_split_passes(self):
        """**1:10 の分割は素通りさせる。** 分割を弾くための検査ではない。"""
        for ratio in (0.1, 0.25, 0.5, 2.0, 4.0, 10.0):
            new = long_frame("1909.T", "2024-03-01", 30, close=1000.0 * ratio)
            self.assertEqual(len(seam_issues(self.old, new)), 0, ratio)

    def test_boundary_is_the_ratio_max(self):
        just_in = long_frame("1909.T", "2024-03-01", 30, close=1000.0 * SEAM_RATIO_MAX)
        just_out = long_frame("1909.T", "2024-03-01", 30,
                              close=1000.0 * SEAM_RATIO_MAX * 1.01)
        self.assertEqual(len(seam_issues(self.old, just_in)), 0)
        self.assertEqual(len(seam_issues(self.old, just_out)), 1)

    def test_zero_volume_run_is_caught(self):
        new = long_frame("1909.T", "2024-03-01", 30, close=1000.0, volume=0.0)
        bad = seam_issues(self.old, new)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad.iloc[0]["kind"], "seam_zero_volume")

    def test_short_zero_volume_run_is_not_caught(self):
        """売買停止は数日で戻る。**短い 0 は止めない。**"""
        new = long_frame("1909.T", "2024-03-01", 5, close=1000.0, volume=0.0)
        self.assertEqual(len(seam_issues(self.old, new)), 0)

    def test_new_listing_has_no_seam(self):
        """既存行が無い銘柄は検査しない。"""
        new = long_frame("9999.T", "2024-03-01", 30, close=1000.0 * 1e6)
        self.assertEqual(len(seam_issues(self.old, new)), 0)

    def test_new_rows_older_than_every_existing_row_have_no_seam(self):
        """バックフィルで履歴を前に伸ばす場合も継ぎ目にならない。"""
        new = long_frame("1909.T", "2020-01-01", 30, close=1000.0 * 1e6)
        self.assertEqual(len(seam_issues(self.old, new)), 0)

    def test_only_the_bad_ticker_is_listed(self):
        old = pd.concat([self.old, long_frame("7203.T", "2024-01-01", 60)],
                        ignore_index=True)
        new = pd.concat([long_frame("1909.T", "2024-03-01", 30, close=1e9),
                         long_frame("7203.T", "2024-03-01", 30, close=1010.0)],
                        ignore_index=True)
        bad = seam_issues(old, new)
        self.assertEqual(list(bad["ticker"]), ["1909.T"])

    def test_empty_inputs(self):
        self.assertEqual(len(seam_issues(pd.DataFrame(), pd.DataFrame())), 0)
        self.assertEqual(len(seam_issues(self.old, pd.DataFrame())), 0)


class UpsertBlocksBadRowsTest(unittest.TestCase):
    """**壊れた行を store に入れない。** これが検査の目的。"""

    def _store(self, tmp):
        return OhlcvStore(Path(tmp) / "store", Path(tmp) / "daily")

    def test_bad_ticker_is_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = self._store(tmp)
            st.save(long_frame("1909.T", "2024-01-01", 60, close=1000.0))
            new = long_frame("1909.T", "2024-03-01", 30, close=1000.0 * 4_399_472)
            merged, added, revisions = st.upsert(new)
            self.assertEqual(len(added), 0)
            self.assertEqual(len(revisions), 0)
            self.assertEqual(len(merged), 60)
            self.assertLess(float(merged["close"].max()), 1e5)
            self.assertEqual(list(st.last_seam_issues["ticker"]), ["1909.T"])

    def test_good_tickers_still_merge_when_one_is_bad(self):
        """**1 銘柄が壊れても他の銘柄は止めない。**"""
        with tempfile.TemporaryDirectory() as tmp:
            st = self._store(tmp)
            st.save(pd.concat([long_frame("1909.T", "2024-01-01", 60),
                               long_frame("7203.T", "2024-01-01", 60)],
                              ignore_index=True))
            new = pd.concat([long_frame("1909.T", "2024-03-01", 30, close=1e9),
                             long_frame("7203.T", "2024-03-01", 30, close=1010.0)],
                            ignore_index=True)
            merged, added, _rev = st.upsert(new)
            self.assertEqual(set(added["ticker"]), {"7203.T"})
            self.assertLess(float(merged[merged["ticker"] == "1909.T"]["close"].max()), 1e5)

    def test_replace_path_is_not_blocked(self):
        """**修復の経路（全履歴再取得）を止めない。** 既存行を先に消すので継ぎ目が無い。"""
        with tempfile.TemporaryDirectory() as tmp:
            st = self._store(tmp)
            st.save(long_frame("1909.T", "2024-01-01", 60, close=1000.0 * 4_399_472))
            fixed = long_frame("1909.T", "2024-01-01", 90, close=1000.0)
            merged, _added, _rev = st.upsert_replace(fixed, ["1909.T"])
            self.assertEqual(len(st.last_seam_issues), 0)
            self.assertEqual(len(merged), 90)
            self.assertAlmostEqual(float(merged["close"].max()), 1000.0)

    def test_healthy_upsert_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = self._store(tmp)
            st.save(long_frame("7203.T", "2024-01-01", 60, close=1000.0))
            merged, added, _rev = st.upsert(
                long_frame("7203.T", "2024-03-01", 30, close=1020.0))
            self.assertEqual(len(st.last_seam_issues), 0)
            self.assertGreater(len(added), 0)
            self.assertGreater(len(merged), 60)

    def test_empty_store_is_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = self._store(tmp)
            merged, added, _rev = st.upsert(long_frame("1909.T", "2024-01-01", 30))
            self.assertEqual(len(merged), 30)
            self.assertEqual(len(st.last_seam_issues), 0)


class SeamRecordTest(unittest.TestCase):
    def test_issues_are_appended_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = OhlcvStore(Path(tmp) / "store", Path(tmp) / "daily")
            bad = pd.DataFrame([{"ticker": "1909.T", "seam_date": pd.Timestamp("2025-01-24"),
                                 "prev_date": pd.Timestamp("2025-01-23"), "new_close": 8.6e9,
                                 "prev_close": 1952.5, "close_ratio": 4399472.0,
                                 "n_new_rows": 399, "kind": "seam_close_ratio"}])
            st.append_seam_issues(bad, pd.Timestamp("2026-09-17"))
            st.append_seam_issues(bad, pd.Timestamp("2026-09-18"))
            out = pd.read_csv(st.seam_path)
            self.assertEqual(len(out), 2)
            self.assertEqual(sorted(out["observed_on"].unique()),
                             ["2026-09-17", "2026-09-18"])

    def test_empty_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = OhlcvStore(Path(tmp) / "store", Path(tmp) / "daily")
            st.append_seam_issues(pd.DataFrame(), pd.Timestamp("2026-09-17"))
            self.assertFalse(st.seam_path.exists())


if __name__ == "__main__":
    unittest.main()
