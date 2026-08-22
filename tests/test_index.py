"""指数データ取得・保存（T-102）。ネットワーク不要。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.store import IDX_TICKER, OhlcvStore, from_long, to_long
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.data.yf_fetch import fetch_index


def _frame(n=70, end="2026-08-21", base=2000.0):
    idx = pd.bdate_range(end=end, periods=n)
    p = np.linspace(base, base * 1.05, n)
    return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p,
                         "Volume": 1e9, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


class FetchIndexTest(unittest.TestCase):
    def test_primary_success(self):
        def downloader(chunk, period):
            self.assertEqual(chunk, ["^TPX"])
            return _frame(320)

        df, used = fetch_index(400, now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                               downloader=downloader, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(used, "^TPX")
        self.assertGreaterEqual(len(df), 300)

    def test_falls_back_when_primary_fails(self):
        def downloader(chunk, period):
            if chunk == ["^TPX"]:
                raise RuntimeError("boom")
            return _frame(320)

        df, used = fetch_index(400, now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                               downloader=downloader, attempts=1, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(used, "^N225")
        self.assertGreaterEqual(len(df), 300)

    def test_empty_when_both_fail(self):
        def downloader(chunk, period):
            raise RuntimeError("boom")

        df, used = fetch_index(400, now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                               downloader=downloader, attempts=1, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(used, "")
        self.assertEqual(len(df), 0)


class SyntheticIndexTest(unittest.TestCase):
    def test_returns_300_or_more_bars(self):
        df = make_synthetic_index(n_bars=400, end=pd.Timestamp("2026-08-21"))
        self.assertGreaterEqual(len(df), 300)
        self.assertTrue((df[["Open", "High", "Low", "Close"]] > 0).all().all())


class IndexStoreRoundtripTest(unittest.TestCase):
    """store に __IDX__ として保存・復元しても RS 計算に未来参照が入らないこと。"""

    def _rs60(self, close: pd.Series, idx: pd.Series) -> pd.Series:
        # DESIGN.md §5 D2: d2_rs60 = ln(Close[T]/Close[T-60]) - ln(IDX[T]/IDX[T-60])
        return np.log(close / close.shift(60)) - np.log(idx / idx.shift(60))

    def test_rs60_recompute_matches_full_period_slice(self):
        end = pd.Timestamp("2026-08-21")
        stock = make_synthetic(["1301.T"], n_bars=400, end=end)["1301.T"]
        index_df = make_synthetic_index(n_bars=400, end=end)

        with TemporaryDirectory() as tmp:
            store = OhlcvStore(Path(tmp) / "store", Path(tmp) / "daily")
            merged, _added, _rev = store.upsert(to_long({"1301.T": stock, IDX_TICKER: index_df}))
            store.save(merged)
            loaded = from_long(store.load())

        close = loaded["1301.T"]["Close"]
        idx = loaded[IDX_TICKER]["Close"]
        rs_full = self._rs60(close, idx)

        for t in (100, 200, 300, len(close) - 1):
            date_t = close.index[t]
            rs_cut = self._rs60(close.loc[:date_t], idx.loc[:date_t])
            full_val = rs_full.loc[date_t]
            cut_val = rs_cut.iloc[-1]
            if pd.isna(full_val):
                self.assertTrue(pd.isna(cut_val))
            else:
                self.assertAlmostEqual(full_val, cut_val, places=10)


if __name__ == "__main__":
    unittest.main()
