import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import _path  # noqa: F401
from stockbot import cli
from stockbot.config import Settings
from stockbot.data.jpx_lists import norm_ticker, normalize_listed
from stockbot.data.synthetic import make_synthetic, synthetic_listed
from stockbot.universe.build import build_universe, liquidity_stats


class TickerTest(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(norm_ticker("1301"), "1301.T")
        self.assertEqual(norm_ticker("130A"), "130A.T")
        self.assertEqual(norm_ticker("8035T"), "8035.T")
        self.assertEqual(norm_ticker("7203.T"), "7203.T")
        self.assertEqual(norm_ticker("25935"), "25935")   # 優先株はそのまま（除外対象）


class ListedTest(unittest.TestCase):
    def test_jpx_columns_and_exclusions(self):
        raw = pd.DataFrame({
            "日付": ["20260731"] * 5,
            "コード": ["1301", "1305", "3283", "25935", "130A"],
            "銘柄名": ["極洋", "ETF", "REIT", "優先株", "新コード"],
            "市場・商品区分": ["プライム（内国株式）", "ETF・ETN",
                           "REIT・ベンチャーファンド・カントリーファンド・インフラファンド",
                           "プライム（内国株式）", "グロース（内国株式）"],
            "33業種コード": [""] * 5, "33業種区分": ["水産・農林業", "", "", "食料品", "医薬品"],
            "17業種コード": [""] * 5, "17業種区分": [""] * 5, "規模コード": [""] * 5, "規模区分": [""] * 5,
        })
        df = normalize_listed(raw)
        eq = df[df["is_equity"]]["ticker"].tolist()
        self.assertListEqual(eq, ["1301.T", "130A.T"])

    def test_seed_csv_columns(self):
        raw = pd.DataFrame({"ticker": ["1301.T"], "name": ["極洋"], "sector": ["水産・農林業"],
                            "industry_big": ["食品"], "market": ["Prime"], "earnings_date": [""]})
        df = normalize_listed(raw)
        self.assertTrue(bool(df["is_equity"].iloc[0]))
        self.assertEqual(df["sector33"].iloc[0], "水産・農林業")


class UniverseTest(unittest.TestCase):
    def test_filters(self):
        listed = synthetic_listed(10)
        ohlcv = make_synthetic(listed["ticker"].tolist(), n_bars=300, seed=1)
        # 1銘柄を履歴不足に、1銘柄を出来高ゼロに
        t0, t1 = listed["ticker"].iloc[0], listed["ticker"].iloc[1]
        ohlcv[t0] = ohlcv[t0].tail(100)
        ohlcv[t1] = ohlcv[t1].assign(Volume=0.0)
        stats = liquidity_stats(ohlcv, 20)
        u = build_universe(listed, stats, min_adv_jpy=1.0, min_price=1.0, min_history_bars=250)
        row0 = u[u["ticker"] == t0].iloc[0]
        row1 = u[u["ticker"] == t1].iloc[0]
        self.assertFalse(bool(row0["passes"]))
        self.assertFalse(bool(row0["ok_history"]))
        self.assertFalse(bool(row1["ok_adv"]))
        self.assertEqual(int(u["passes"].sum()), 8)


class ManualExclusionWiringTest(unittest.TestCase):
    """データ品質による手動除外リスト（2026-08-27 追加）が step_universe に反映されること。
    9900.T（先出し分割適用でadjust.pyの自動検出をすり抜けたケース）への対応。"""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._env_names = ("SCREEN_DRYRUN", "DATA_DIR")
        self._saved = {k: os.environ.get(k) for k in self._env_names}
        os.environ["SCREEN_DRYRUN"] = "1"
        os.environ["DATA_DIR"] = self._tmp.name
        self.cfg = Settings.from_env()
        self.cfg.ensure_dirs()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_ticker_in_manual_exclusions_file_fails_universe(self):
        listed = synthetic_listed(10)
        excluded_ticker = listed["ticker"].iloc[0]
        ohlcv = make_synthetic(listed["ticker"].tolist(), n_bars=300, seed=1)

        (self.cfg.reference_dir / "manual_exclusions.csv").write_text(
            f"ticker,until,reason\n{excluded_ticker},2099-01-01,test exclusion\n",
            encoding="utf-8")

        u = cli.step_universe(self.cfg, listed, ohlcv=ohlcv, issues=None, log=lambda m: None)
        row = u[u["ticker"] == excluded_ticker].iloc[0]
        self.assertFalse(bool(row["ok_split"]))
        self.assertFalse(bool(row["passes"]))

    def test_expired_manual_exclusion_does_not_affect_universe(self):
        listed = synthetic_listed(10)
        ticker = listed["ticker"].iloc[0]
        ohlcv = make_synthetic(listed["ticker"].tolist(), n_bars=300, seed=1)

        (self.cfg.reference_dir / "manual_exclusions.csv").write_text(
            f"ticker,until,reason\n{ticker},2000-01-01,expired\n", encoding="utf-8")

        u = cli.step_universe(self.cfg, listed, ohlcv=ohlcv, issues=None, log=lambda m: None)
        row = u[u["ticker"] == ticker].iloc[0]
        self.assertTrue(bool(row["ok_split"]))

    def test_manual_exclusion_merges_with_automatic_suspected_split_issues(self):
        listed = synthetic_listed(10)
        manual_ticker = listed["ticker"].iloc[0]
        auto_ticker = listed["ticker"].iloc[1]
        ohlcv = make_synthetic(listed["ticker"].tolist(), n_bars=300, seed=1)

        (self.cfg.reference_dir / "manual_exclusions.csv").write_text(
            f"ticker,until,reason\n{manual_ticker},2099-01-01,test exclusion\n", encoding="utf-8")
        issues = pd.DataFrame({
            "ticker": [auto_ticker], "date": [pd.Timestamp("2026-08-01")],
            "kind": ["suspected_unrecorded_split"], "ratio": [2.0],
            "observed": [2.0], "action": ["flag_only"],
        })

        u = cli.step_universe(self.cfg, listed, ohlcv=ohlcv, issues=issues, log=lambda m: None)
        self.assertFalse(bool(u[u["ticker"] == manual_ticker].iloc[0]["ok_split"]))
        self.assertFalse(bool(u[u["ticker"] == auto_ticker].iloc[0]["ok_split"]))


if __name__ == "__main__":
    unittest.main()
