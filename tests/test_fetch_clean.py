"""取得層の清掃ロジック（ネットワーク不要）。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.yf_fetch import COLS, clean_frame, fetch_ohlcv, flatten_single


def _frame(n=70, end="2026-08-21"):
    idx = pd.bdate_range(end=end, periods=n)
    p = np.linspace(1000, 1100, n)
    return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p,
                         "Volume": 1e5, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


class CleanFrameTest(unittest.TestCase):
    def test_drops_today_before_close_and_keeps_after(self):
        df = _frame(end="2026-08-21")  # 金曜
        before = pd.Timestamp("2026-08-21 10:00", tz="Asia/Tokyo")
        after = pd.Timestamp("2026-08-21 15:31", tz="Asia/Tokyo")
        self.assertEqual(clean_frame(df, before).index[-1], pd.Timestamp("2026-08-20"))
        self.assertEqual(clean_frame(df, after).index[-1], pd.Timestamp("2026-08-21"))

    def test_drops_rows_with_nan_price(self):
        df = _frame()
        df.iloc[5, df.columns.get_loc("Close")] = np.nan
        out = clean_frame(df, pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"))
        self.assertEqual(len(out), len(df) - 1)
        self.assertListEqual(list(out.columns), COLS)

    def test_tz_aware_index_is_normalized(self):
        df = _frame()
        df.index = df.index.tz_localize("Asia/Tokyo")
        out = clean_frame(df, pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"))
        self.assertIsNone(out.index.tz)

    def test_flatten_single_multiindex(self):
        df = _frame()
        mi = pd.concat({"1301.T": df}, axis=1)
        self.assertListEqual(list(flatten_single(mi, "1301.T").columns), list(df.columns))


class FetchLoopTest(unittest.TestCase):
    def test_retry_rounds_recover_failures(self):
        """1巡目で一部失敗 → 小さいチャンクで回収されること。"""
        tickers = [f"{1300 + i}.T" for i in range(12)]
        calls = []

        def downloader(chunk, period):
            calls.append(len(chunk))
            if len(chunk) > 10:           # 大チャンクは半分だけ返す
                chunk = chunk[:6]
            return pd.concat({t: _frame() for t in chunk}, axis=1)

        out, meta = fetch_ohlcv(tickers, 400, deadline_sec=60,
                                now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                                downloader=downloader, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(meta["data_ok"], 12)
        self.assertEqual(meta["failed"], [])
        self.assertGreater(len(calls), 1)

    def test_short_history_is_not_retried_forever(self):
        def downloader(chunk, period):
            return pd.concat({t: _frame(n=10) for t in chunk}, axis=1)

        out, meta = fetch_ohlcv(["1301.T"], 400, deadline_sec=60,
                                now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                                downloader=downloader, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(meta["data_ok"], 0)
        self.assertEqual(meta["short"], ["1301.T"])
        self.assertEqual(meta["failed"], [])


if __name__ == "__main__":
    unittest.main()
