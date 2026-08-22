"""Settings.from_env の既定値。特に DRYRUN と本番のディレクトリ分離（2026-08-22 の
データ混入事故の再発防止）を固定するテスト。"""
import os
import unittest

from . import _path  # noqa: F401
from stockbot.config import Settings


class SettingsEnvTest(unittest.TestCase):
    _PARAM_ENV_NAMES = (
        "SCREEN_DRYRUN", "DATA_DIR",
        "K", "PULLBACK_MAX_DAYS", "PULLBACK_MAX_DEPTH", "SHALLOW_DEPTH",
        "DEEP_DEPTH", "POOL_DAYS", "SECTOR_CAP", "CORR_THRESHOLD",
        "TOP_N", "LABEL_N",
    )

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._PARAM_ENV_NAMES}
        for k in self._PARAM_ENV_NAMES:
            os.environ.pop(k, None)

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

    def test_registered_params_match_design_defaults(self):
        # DESIGN.md §12 の事前登録パラメータの既定値
        cfg = Settings.from_env()
        self.assertEqual(cfg.k, 3)
        self.assertEqual(cfg.pullback_max_days, 25)
        self.assertAlmostEqual(cfg.pullback_max_depth, 0.25)
        self.assertAlmostEqual(cfg.shallow_depth, 0.03)
        self.assertAlmostEqual(cfg.deep_depth, 0.10)
        self.assertEqual(cfg.pool_days, 20)
        self.assertEqual(cfg.sector_cap, 3)
        self.assertAlmostEqual(cfg.corr_threshold, 0.8)
        self.assertEqual(cfg.top_n, 10)
        self.assertEqual(cfg.label_n, 15)

    def test_registered_params_overridable_by_env(self):
        os.environ["K"] = "2"
        os.environ["PULLBACK_MAX_DAYS"] = "20"
        os.environ["PULLBACK_MAX_DEPTH"] = "0.20"
        os.environ["SHALLOW_DEPTH"] = "0.02"
        os.environ["DEEP_DEPTH"] = "0.12"
        os.environ["POOL_DAYS"] = "25"
        os.environ["SECTOR_CAP"] = "4"
        os.environ["CORR_THRESHOLD"] = "0.75"
        os.environ["TOP_N"] = "12"
        os.environ["LABEL_N"] = "10"
        cfg = Settings.from_env()
        self.assertEqual(cfg.k, 2)
        self.assertEqual(cfg.pullback_max_days, 20)
        self.assertAlmostEqual(cfg.pullback_max_depth, 0.20)
        self.assertAlmostEqual(cfg.shallow_depth, 0.02)
        self.assertAlmostEqual(cfg.deep_depth, 0.12)
        self.assertEqual(cfg.pool_days, 25)
        self.assertEqual(cfg.sector_cap, 4)
        self.assertAlmostEqual(cfg.corr_threshold, 0.75)
        self.assertEqual(cfg.top_n, 12)
        self.assertEqual(cfg.label_n, 10)


if __name__ == "__main__":
    unittest.main()
