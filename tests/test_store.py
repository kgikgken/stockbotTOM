import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.store import OhlcvStore, from_long, to_long


def _ohlcv(n=5, end="2026-08-21", close0=1000.0):
    idx = pd.bdate_range(end=end, periods=n)
    p = close0 + np.arange(n)
    df = pd.DataFrame({"Open": p, "High": p + 1, "Low": p - 1, "Close": p, "Volume": 1000.0,
                       "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    df.index.name = "Date"
    return df


class StoreTest(unittest.TestCase):
    def test_roundtrip_and_increments(self):
        with tempfile.TemporaryDirectory() as d:
            st = OhlcvStore(Path(d) / "store", Path(d) / "daily")
            first = {"1301.T": _ohlcv(5, "2026-08-20")}
            merged, added, rev = st.upsert(to_long(first))
            st.save(merged)
            self.assertEqual(len(added), 5)
            self.assertEqual(len(rev), 0)
            st.write_daily_increments(added)

            # 翌日: 1本追加 + 既存1本の終値が改訂
            second = {"1301.T": _ohlcv(6, "2026-08-21")}
            second["1301.T"].iloc[0, second["1301.T"].columns.get_loc("Close")] = 1500.0
            merged2, added2, rev2 = st.upsert(to_long(second))
            st.save(merged2)
            files = st.write_daily_increments(added2)
            st.append_revisions(rev2, pd.Timestamp("2026-08-22"))

            self.assertEqual(len(added2), 1)
            self.assertEqual(len(rev2), 1)
            self.assertEqual(len(merged2), 6)
            self.assertTrue((Path(d) / "daily" / "2026-08-21.csv.gz").exists())
            self.assertTrue(st.revisions_path.exists())
            back = from_long(st.load())
            self.assertEqual(len(back["1301.T"]), 6)
            self.assertEqual(float(back["1301.T"]["Close"].iloc[0]), 1500.0)
            self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
