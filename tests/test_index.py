"""指数データ取得・保存（T-102）。ネットワーク不要。"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot import cli
from stockbot.config import Settings
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

    def test_falls_back_to_topix_etf_when_raw_index_fails(self):
        # ^TPXが取れない場合、日経225ではなくTOPIX連動ETF(1306.T)を先に試す
        # （実際に^TPXがyfinanceでほぼ常に取得不能なことを2026-08-24に確認済み）
        def downloader(chunk, period):
            if chunk == ["^TPX"]:
                raise RuntimeError("boom")
            return _frame(320)

        df, used = fetch_index(400, now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                               downloader=downloader, attempts=1, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(used, "1306.T")
        self.assertGreaterEqual(len(df), 300)

    def test_falls_back_to_nikkei_when_both_topix_sources_fail(self):
        def downloader(chunk, period):
            if chunk in (["^TPX"], ["1306.T"]):
                raise RuntimeError("boom")
            return _frame(320)

        df, used = fetch_index(400, now_jst=pd.Timestamp("2026-08-22 09:00", tz="Asia/Tokyo"),
                               downloader=downloader, attempts=1, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(used, "^N225")
        self.assertGreaterEqual(len(df), 300)

    def test_empty_when_all_candidates_fail(self):
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


class StepIndexMetaTest(unittest.TestCase):
    """cli.step_index が store/index_meta.json に実際に使ったティッカーを記録すること
    （TOPIXが取得できず日経225にフォールバックした場合、L1レポート(T-405)に
    明記するために validation.report が参照する）。"""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._env_names = ("SCREEN_DRYRUN", "DATA_DIR", "HISTORY_DAYS")
        self._saved = {k: os.environ.get(k) for k in self._env_names}
        os.environ["DATA_DIR"] = self._tmp.name
        os.environ["HISTORY_DAYS"] = "120"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _meta(self, cfg):
        return json.loads((cfg.store_dir / "index_meta.json").read_text(encoding="utf-8"))

    def test_dryrun_is_labeled_synthetic_and_not_fallback(self):
        os.environ["SCREEN_DRYRUN"] = "1"
        cfg = Settings.from_env()
        cli.step_index(cfg, log=lambda m: None)
        meta = self._meta(cfg)
        self.assertEqual(meta["label"], "合成(DRYRUN)")
        self.assertFalse(meta["is_fallback"])

    def test_primary_topix_success_is_not_fallback(self):
        os.environ.pop("SCREEN_DRYRUN", None)
        cfg = Settings.from_env()
        df = make_synthetic_index(n_bars=120, end=pd.Timestamp("2026-08-21"))
        with patch("stockbot.cli.fetch_index", return_value=(df, "^TPX")):
            cli.step_index(cfg, log=lambda m: None)
        meta = self._meta(cfg)
        self.assertEqual(meta, {"ticker": "^TPX", "label": "TOPIX", "is_fallback": False})

    def test_fallback_to_nikkei_is_recorded(self):
        os.environ.pop("SCREEN_DRYRUN", None)
        cfg = Settings.from_env()
        df = make_synthetic_index(n_bars=120, end=pd.Timestamp("2026-08-21"))
        with patch("stockbot.cli.fetch_index", return_value=(df, "^N225")):
            cli.step_index(cfg, log=lambda m: None)
        meta = self._meta(cfg)
        self.assertEqual(meta, {"ticker": "^N225", "label": "日経225", "is_fallback": True})

    def test_topix_etf_proxy_is_not_flagged_as_fallback(self):
        # 1306.T はTOPIXを追跡するETFで別指数への切り替えではないため、
        # is_fallback=False とし、L1レポートに警告文を出させない
        os.environ.pop("SCREEN_DRYRUN", None)
        cfg = Settings.from_env()
        df = make_synthetic_index(n_bars=120, end=pd.Timestamp("2026-08-21"))
        with patch("stockbot.cli.fetch_index", return_value=(df, "1306.T")):
            cli.step_index(cfg, log=lambda m: None)
        meta = self._meta(cfg)
        self.assertEqual(meta, {"ticker": "1306.T", "label": "TOPIX(ETF代替: 1306.T)",
                                "is_fallback": False})

    def test_total_failure_does_not_overwrite_previous_meta(self):
        """前回成功時の記録を、今回が両方失敗（前回値のまま）で上書きしない。"""
        os.environ.pop("SCREEN_DRYRUN", None)
        cfg = Settings.from_env()
        df = make_synthetic_index(n_bars=120, end=pd.Timestamp("2026-08-21"))
        with patch("stockbot.cli.fetch_index", return_value=(df, "^TPX")):
            cli.step_index(cfg, log=lambda m: None)
        before = self._meta(cfg)

        empty = pd.DataFrame(columns=df.columns)
        with patch("stockbot.cli.fetch_index", return_value=(empty, "")):
            cli.step_index(cfg, log=lambda m: None)
        after = self._meta(cfg)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
