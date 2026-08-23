"""バックフィル（T-103）。中断再開（本数基準）と取得率の記録。ネットワーク不要（DRYRUN）。"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import _path  # noqa: F401
from stockbot import cli
from stockbot.config import Settings
from stockbot.data.store import OhlcvStore, to_long
from stockbot.data.synthetic import make_synthetic, synthetic_listed


class BackfillResumeTest(unittest.TestCase):
    END = pd.Timestamp("2026-08-21")

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._env_names = ("SCREEN_DRYRUN", "DATA_DIR", "HISTORY_DAYS")
        self._saved = {k: os.environ.get(k) for k in self._env_names}
        os.environ["SCREEN_DRYRUN"] = "1"
        os.environ["DATA_DIR"] = self._tmp.name
        os.environ["HISTORY_DAYS"] = "120"
        self.cfg = Settings.from_env()
        self.threshold = int(self.cfg.history_days * 0.9)  # 108

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_resume_skips_tickers_with_sufficient_bars(self):
        listed = synthetic_listed(60)
        eq = listed[listed["is_equity"].astype(bool)]["ticker"].tolist()
        self.assertEqual(len(eq), 60)

        # 事前に半分だけ、目標本数(120)ぴったりで store に入れておく（中断後の状態を模す）
        preloaded = eq[:20]
        store = OhlcvStore(self.cfg.store_dir, self.cfg.daily_dir)
        seed_ohlcv = make_synthetic(preloaded, n_bars=120, end=self.END)
        merged, _added, _rev = store.upsert(to_long(seed_ohlcv))
        store.save(merged)

        meta1 = cli.step_backfill(self.cfg, log=lambda m: None)
        self.assertEqual(meta1["universe_total"], 60)
        self.assertEqual(meta1["bar_threshold"], self.threshold)
        self.assertEqual(meta1["already_done"], 20)
        self.assertEqual(meta1["data_total"], 40)         # 今回の取得対象は残り40件のみ
        self.assertEqual(meta1["newly_done"], 40)
        self.assertEqual(meta1["cumulative_done"], 60)
        self.assertAlmostEqual(meta1["completion_rate"], 1.0)

        meta_path = self.cfg.store_dir / "backfill_meta.json"
        self.assertTrue(meta_path.exists())
        saved = json.loads(meta_path.read_text())
        self.assertEqual(saved["cumulative_done"], 60)

        # 再実行: 全銘柄が本数基準を満たしているので、追加取得は発生しない
        meta2 = cli.step_backfill(self.cfg, log=lambda m: None)
        self.assertEqual(meta2["already_done"], 60)
        self.assertEqual(meta2["data_total"], 0)
        self.assertEqual(meta2["newly_done"], 0)
        self.assertEqual(meta2["cumulative_done"], 60)
        self.assertAlmostEqual(meta2["completion_rate"], 1.0)

    def test_ticker_present_but_short_is_still_a_target(self):
        """store に存在していても本数が目標(120)の90%(108)未満なら再取得対象に含まれること。"""
        listed = synthetic_listed(60)
        eq = listed[listed["is_equity"].astype(bool)]["ticker"].tolist()
        short_ticker = eq[0]

        store = OhlcvStore(self.cfg.store_dir, self.cfg.daily_dir)
        # 400本目標のうち400本ではなく60本しか無い状態（本数不足）を再現
        seed_ohlcv = make_synthetic([short_ticker], n_bars=60, end=self.END)
        merged, _added, _rev = store.upsert(to_long(seed_ohlcv))
        store.save(merged)

        before_counts = store.load().groupby("ticker")["date"].count()
        self.assertEqual(int(before_counts[short_ticker]), 60)
        self.assertLess(60, self.threshold)  # 前提: 60 < 108 で不足であること

        meta = cli.step_backfill(self.cfg, log=lambda m: None)
        # 存在するが本数不足だった1銘柄も、他59銘柄と合わせて対象に入る
        self.assertEqual(meta["already_done"], 0)
        self.assertEqual(meta["data_total"], 60)

        after_counts = OhlcvStore(self.cfg.store_dir, self.cfg.daily_dir).load().groupby("ticker")["date"].count()
        self.assertGreaterEqual(int(after_counts[short_ticker]), self.threshold)
        self.assertEqual(meta["cumulative_done"], 60)
        self.assertAlmostEqual(meta["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
