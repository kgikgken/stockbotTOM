"""scripts/verify_chunk_coverage.py の突合ロジックのテスト（診断用スクリプト。
2026-08-29 指示: Actionsのキャッシュ上書き等で静かに欠けた日が無いかを確認する）。"""
from __future__ import annotations

import gzip
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_chunk_coverage import check_coverage, saved_dates  # noqa: E402


def _touch_csv_gz(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        f.write("ticker\n")


class SavedDatesTest(unittest.TestCase):
    def test_parses_dates_from_filenames(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _touch_csv_gz(tmp / "a_prime_2024-01-02.csv.gz")
            _touch_csv_gz(tmp / "a_prime_2024-01-03.csv.gz")
            dates = saved_dates(tmp, "a_prime")
            self.assertEqual(dates, {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")})

    def test_missing_dir_returns_empty(self):
        dates = saved_dates(Path("/does/not/exist"), "replay")
        self.assertEqual(dates, set())

    def test_ignores_non_matching_prefix(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _touch_csv_gz(tmp / "replay_2024-01-02.csv.gz")
            dates = saved_dates(tmp, "a_prime")
            self.assertEqual(dates, set())


class CheckCoverageTest(unittest.TestCase):
    def test_detects_missing_and_extra_dates(self):
        dates = pd.bdate_range("2018-07-13", "2018-07-18")
        phantom_day = pd.Timestamp("2018-07-16")  # 海の日（休場、幻日）
        real_days = dates.drop(phantom_day)
        # 10銘柄中2銘柄だけ幻日にもデータを持つ（実データで確認済みのパターン）
        ohlcv = {}
        for i in range(10):
            idx = dates if i < 2 else real_days
            ohlcv[f"T{i}"] = pd.DataFrame({"Close": 1.0}, index=idx)

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # real_days のうち1日をわざと保存し忘れる（不足の検出）
            missing_day = real_days[0]
            for d in real_days:
                if d == missing_day:
                    continue
                _touch_csv_gz(tmp / f"replay_{d.date().isoformat()}.csv.gz")
            # 幻日をわざと保存してしまっているケース（余剰の検出）
            _touch_csv_gz(tmp / f"replay_{phantom_day.date().isoformat()}.csv.gz")

            result = check_coverage(dates[0], dates[-1], tmp, "replay", ohlcv, log=lambda *a: None)

        self.assertEqual(result["n_expected"], len(real_days))
        self.assertEqual(result["n_missing"], 1)
        self.assertIn(missing_day.date().isoformat(), result["missing"])
        self.assertEqual(result["n_extra"], 1)
        self.assertIn(phantom_day.date().isoformat(), result["extra"])

    def test_no_gaps_reports_zero(self):
        dates = pd.bdate_range("2018-07-13", "2018-07-17")
        ohlcv = {f"T{i}": pd.DataFrame({"Close": 1.0}, index=dates) for i in range(10)}
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for d in dates:
                _touch_csv_gz(tmp / f"a_prime_{d.date().isoformat()}.csv.gz")
            result = check_coverage(dates[0], dates[-1], tmp, "a_prime", ohlcv, log=lambda *a: None)
        self.assertEqual(result["n_missing"], 0)
        self.assertEqual(result["n_extra"], 0)
        self.assertEqual(result["n_expected"], result["n_saved_in_range"])


if __name__ == "__main__":
    unittest.main()
