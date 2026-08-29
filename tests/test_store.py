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


class UpsertReplaceTest(unittest.TestCase):
    """T-402: 全履歴再取得はマージだと取得ウィンドウの外の古い行が取り残されて
    段差が生じる（9900.Tの2016-02-09境界で実データ確認済み）。upsert_replaceは
    対象銘柄の既存行を先に削除してから統合し、fetchできた範囲だけが残ることを
    確認する。"""

    def test_replace_drops_dates_outside_new_fetch_window(self):
        with tempfile.TemporaryDirectory() as d:
            st = OhlcvStore(Path(d) / "store", Path(d) / "daily")
            # 初回: 10本(2026-08-07〜2026-08-20)取得したことにする
            wide = {"9900.T": _ohlcv(10, "2026-08-20")}
            merged, _added, _rev = st.upsert(to_long(wide))
            st.save(merged)
            self.assertEqual(len(merged), 10)

            # 全履歴再取得で直近5本だけ取れた場合（古い5本は取得窓の外）
            narrow = {"9900.T": _ohlcv(5, "2026-08-20", close0=2000.0)}
            merged2, _added2, _rev2 = st.upsert_replace(to_long(narrow), ["9900.T"])
            self.assertEqual(len(merged2), 5)  # 古い5本は残らない(マージなら10本のまま)
            self.assertEqual(float(merged2["close"].min()), 2000.0)

    def test_replace_leaves_other_tickers_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            st = OhlcvStore(Path(d) / "store", Path(d) / "daily")
            wide = {"9900.T": _ohlcv(10, "2026-08-20"), "1301.T": _ohlcv(10, "2026-08-20")}
            merged, _added, _rev = st.upsert(to_long(wide))
            st.save(merged)

            narrow = {"9900.T": _ohlcv(5, "2026-08-20", close0=2000.0)}
            merged2, _added2, _rev2 = st.upsert_replace(to_long(narrow), ["9900.T"])
            self.assertEqual(len(merged2[merged2["ticker"] == "9900.T"]), 5)
            self.assertEqual(len(merged2[merged2["ticker"] == "1301.T"]), 10)  # 対象外は無傷

    def test_replace_with_empty_ticker_list_behaves_like_upsert(self):
        with tempfile.TemporaryDirectory() as d:
            st = OhlcvStore(Path(d) / "store", Path(d) / "daily")
            wide = {"9900.T": _ohlcv(10, "2026-08-20")}
            merged, _added, _rev = st.upsert(to_long(wide))
            st.save(merged)

            narrow = {"9900.T": _ohlcv(5, "2026-08-20", close0=2000.0)}
            merged2, _added2, _rev2 = st.upsert_replace(to_long(narrow), [])
            self.assertEqual(len(merged2), 10)  # 対象銘柄指定なし=通常のマージと同じ


if __name__ == "__main__":
    unittest.main()
