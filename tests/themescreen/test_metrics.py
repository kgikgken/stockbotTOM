import csv
import math
import unittest
import pandas as pd

from .. import _path  # noqa: F401
from themescreen.config import UNIVERSE_DIR
from themescreen.metrics import (
    PROCESSES, Computed, cutoff, latest_annual, market_cap, median_trading_value,
    next_period_end_passed,
)


def _hist(n: int, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "close": [100.0] * n,
                         "volume": [float(i + 1) for i in range(n)]})


class TestMedianTradingValue(unittest.TestCase):
    def test_last_60_sessions(self) -> None:
        h = _hist(80)
        # last 60 volumes are 21..80 -> median 50.5, close 100
        r = median_trading_value(h, asof="2099-01-01")
        self.assertAlmostEqual(r.value, 100.0 * 50.5)

    def test_rows_after_asof_ignored(self) -> None:
        h = _hist(80)
        asof = h["date"].iloc[69]  # 70 rows on or before asof -> volumes 11..70
        r = median_trading_value(h, asof=asof)
        self.assertAlmostEqual(r.value, 100.0 * 40.5)
        # changing data after asof does not change the result
        h2 = h.copy()
        h2.loc[h2.index[70:], "volume"] = 1e12
        self.assertEqual(median_trading_value(h2, asof=asof).value, r.value)

    def test_fewer_than_60_not_computable(self) -> None:
        r = median_trading_value(_hist(59), asof="2099-01-01")
        self.assertFalse(r.ok)
        self.assertIn("59", r.reason)

    def test_nan_rows_dropped(self) -> None:
        h = _hist(60)
        h.loc[h.index[0], "close"] = float("nan")
        self.assertFalse(median_trading_value(h, asof="2099-01-01").ok)

    def test_empty(self) -> None:
        self.assertFalse(median_trading_value(pd.DataFrame(), asof="2099-01-01").ok)


class TestLatestAnnual(unittest.TestCase):
    def test_picks_latest(self) -> None:
        d, r = latest_annual({"2025-03-31": -5.0, "2026-03-31": 7.0})
        self.assertEqual(d, "2026-03-31")
        self.assertEqual(r.value, 7.0)

    def test_latest_missing_does_not_fall_back(self) -> None:
        d, r = latest_annual({"2025-03-31": 5.0, "2026-03-31": float("nan")})
        self.assertEqual(d, "2026-03-31")
        self.assertFalse(r.ok)

    def test_none(self) -> None:
        self.assertFalse(latest_annual(None)[1].ok)


class TestMarketCap(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertEqual(market_cap({"currency": "JPY", "marketCap": 5e11, "quoteType": "EQUITY"}).value, 5e11)

    def test_missing_or_wrong_currency(self) -> None:
        self.assertFalse(market_cap({"currency": "JPY", "quoteType": "EQUITY"}).ok)
        self.assertFalse(market_cap({"currency": "USD", "marketCap": 1.0, "quoteType": "EQUITY"}).ok)
        self.assertFalse(market_cap(None).ok)
        self.assertIn("返さない", market_cap({"currency": None, "quoteType": "NONE"}).reason)
        self.assertFalse(market_cap({"currency": "JPY", "marketCap": math.nan, "quoteType": "EQUITY"}).ok)


class TestNextPeriodEndPassed(unittest.TestCase):
    def test_passed(self) -> None:
        self.assertEqual(next_period_end_passed("2025-06-30", "2026-10-02"), ("2026-06-30", 94))

    def test_not_passed(self) -> None:
        self.assertIsNone(next_period_end_passed("2026-03-31", "2026-10-02"))
        self.assertIsNone(next_period_end_passed("2025-12-31", "2026-10-02"))
        self.assertIsNone(next_period_end_passed(None, "2026-10-02"))

    def test_same_day_counts_as_passed(self) -> None:
        self.assertEqual(next_period_end_passed("2025-10-02", "2026-10-02"), ("2026-10-02", 0))

    def test_february_month_end(self) -> None:
        self.assertEqual(next_period_end_passed("2027-02-28", "2028-03-01"), ("2028-02-29", 1))


class TestCutoff(unittest.TestCase):
    def test_boundary_100m_is_kept(self) -> None:
        self.assertEqual(cutoff(Computed(1e8), Computed(1.0)), ("掲載", []))

    def test_below_100m_excluded(self) -> None:
        self.assertEqual(cutoff(Computed(1e8 - 1), Computed(1.0))[1], ["売買代金1億円未満"])

    def test_zero_operating_income_is_not_loss(self) -> None:
        self.assertEqual(cutoff(Computed(2e8), Computed(0.0))[0], "掲載")

    def test_loss_excluded(self) -> None:
        self.assertEqual(cutoff(Computed(2e8), Computed(-1.0)), ("除外", ["直近期営業赤字"]))

    def test_both(self) -> None:
        self.assertEqual(cutoff(Computed(1.0), Computed(-1.0))[1],
                         ["売買代金1億円未満", "直近期営業赤字"])

    def test_missing_input(self) -> None:
        self.assertEqual(cutoff(Computed(None, "x"), Computed(1.0))[0], "判定不能")
        self.assertEqual(cutoff(Computed(2e8), Computed(None, "x"))[0], "判定不能")
        # a computable cutoff that triggers still excludes
        self.assertEqual(cutoff(Computed(None, "x"), Computed(-1.0))[0], "除外")


class TestUniverse(unittest.TestCase):
    def test_universe_file(self) -> None:
        with open(UNIVERSE_DIR / "semicon_frontend.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        codes = [r["code"] for r in rows]
        self.assertEqual(len(codes), len(set(codes)))
        for r in rows:
            self.assertRegex(r["code"], r"^[0-9][0-9A-Z]{3}$")
            self.assertIn(r["process"], PROCESSES)
            self.assertTrue(r["name"] and r["basis"])


if __name__ == "__main__":
    unittest.main()
