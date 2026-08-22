"""Settings.from_env の既定値。特に DRYRUN と本番のディレクトリ分離（2026-08-22 の
データ混入事故の再発防止）を固定するテスト。"""
import os
import unittest

from . import _path  # noqa: F401
from stockbot.config import Settings


class SettingsEnvTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("SCREEN_DRYRUN", "DATA_DIR")}
        os.environ.pop("SCREEN_DRYRUN", None)
        os.environ.pop("DATA_DIR", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_dryrun_defaults_to_isolated_dir(self):
        os.environ["SCREEN_DRYRUN"] = "1"
        cfg = Settings.from_env()
        self.assertTrue(cfg.dryrun)
        self.assertEqual(str(cfg.data_dir), "data-dryrun")

    def test_real_run_defaults_to_data(self):
        cfg = Settings.from_env()
        self.assertFalse(cfg.dryrun)
        self.assertEqual(str(cfg.data_dir), "data")

    def test_explicit_data_dir_overrides_dryrun_default(self):
        os.environ["SCREEN_DRYRUN"] = "1"
        os.environ["DATA_DIR"] = "/tmp/custom"
        cfg = Settings.from_env()
        self.assertEqual(str(cfg.data_dir), "/tmp/custom")

    def test_dryrun_and_real_never_share_a_path(self):
        os.environ["SCREEN_DRYRUN"] = "1"
        dryrun_dir = Settings.from_env().data_dir
        os.environ["SCREEN_DRYRUN"] = "0"
        real_dir = Settings.from_env().data_dir
        self.assertNotEqual(dryrun_dir, real_dir)


if __name__ == "__main__":
    unittest.main()
