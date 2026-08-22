"""バックフィル（T-103）。中断再開と取得率の記録。ネットワーク不要（DRYRUN）。"""
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
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._env_names = ("SCREEN_DRYRUN", "DATA_DIR", "HISTORY_DAYS")
        self._saved = {k: os.environ.get(k) for k in self._env_names}
        os.environ["SCREEN_DRYRUN"] = "1"
        os.environ["DATA_DIR"] = self._tmp.name
        os.environ["HISTORY_DAYS"] = "120"
        self.cfg = Settings.from_env()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_resume_skips_tickers_already_in_store(self):
        listed = synthetic_listed(60)
        eq = listed[listed["is_equity"].astype(bool)]["ticker"].tolist()
        self.assertEqual(len(eq), 60)

        # 事前に半分だけ store に入れておく（中断後の状態を模す）
        preloaded = eq[:20]
        store = OhlcvStore(self.cfg.store_dir, self.cfg.daily_dir)
        seed_ohlcv = make_synthetic(preloaded, n_bars=120, end=pd.Timestamp("2026-08-21"))
        merged, _added, _rev = store.upsert(to_long(seed_ohlcv))
        store.save(merged)

        meta1 = cli.step_backfill(self.cfg, log=lambda m: None)
        self.assertEqual(meta1["universe_total"], 60)
        self.assertEqual(meta1["already_done"], 20)
        self.assertEqual(meta1["data_total"], 40)         # 今回の取得対象は残り40件のみ
        self.assertEqual(meta1["newly_done"], 40)
        self.assertEqual(meta1["cumulative_done"], 60)
        self.assertAlmostEqual(meta1["completion_rate"], 1.0)

        store_after1 = OhlcvStore(self.cfg.store_dir, self.cfg.daily_dir)
        self.assertEqual(len(set(store_after1.load()["ticker"].unique()) & set(eq)), 60)

        meta_path = self.cfg.store_dir / "backfill_meta.json"
        self.assertTrue(meta_path.exists())
        saved = json.loads(meta_path.read_text())
        self.assertEqual(saved["cumulative_done"], 60)

        # 再実行: 全銘柄が既に store にあるので、追加取得は発生しない
        meta2 = cli.step_backfill(self.cfg, log=lambda m: None)
        self.assertEqual(meta2["already_done"], 60)
        self.assertEqual(meta2["data_total"], 0)
        self.assertEqual(meta2["newly_done"], 0)
        self.assertEqual(meta2["cumulative_done"], 60)
        self.assertAlmostEqual(meta2["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
