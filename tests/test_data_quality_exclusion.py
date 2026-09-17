"""データ品質による除外（2026-09-17・1909.T と 8303.T）。

固定するのは 4 つ。**T-402 の事前登録セットを字面のまま残すこと**、**除外が
ユニバース・ベンチマーク・パターン検出のすべてに効くこと**、**取得データが壊れて
いれば置換しないこと**、**store から行を消さないこと**。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.adjust import FETCH_ABSURD_CLOSE, fetch_sanity_issues
from stockbot.universe.build import apply_filters
from stockbot.validation import pattern_replay as replay_mod
from stockbot.validation.labels import universe_benchmark_returns
from stockbot.validation.layer1 import (
    DATA_QUALITY_EXCLUDED_2026_09,
    DATA_QUALITY_EXCLUDED_T402,
    DATA_QUALITY_EXCLUDED_TICKERS,
    exclude_data_quality_tickers,
)

BAD = "1909.T"
BAD2 = "8303.T"


def frame(n=40, close=1000.0, start="2024-01-01", volume=1e6):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": volume}, index=idx)


class ConstantTest(unittest.TestCase):
    def test_t402_set_is_untouched(self):
        """**事前登録セットは字面のまま。** 「適用時の変更禁止」と書いてある。"""
        self.assertEqual(DATA_QUALITY_EXCLUDED_T402, frozenset({
            "1364.T", "3477.T", "4316.T", "5103.T", "6731.T",
            "6834.T", "7649.T", "7877.T", "7983.T", "9900.T"}))

    def test_new_tickers_are_a_separate_set(self):
        self.assertEqual(DATA_QUALITY_EXCLUDED_2026_09, frozenset({BAD, BAD2}))
        self.assertFalse(DATA_QUALITY_EXCLUDED_T402 & DATA_QUALITY_EXCLUDED_2026_09)

    def test_combined_set_is_what_everything_uses(self):
        self.assertEqual(DATA_QUALITY_EXCLUDED_TICKERS,
                         DATA_QUALITY_EXCLUDED_T402 | DATA_QUALITY_EXCLUDED_2026_09)
        for t in (BAD, BAD2, "9900.T"):
            self.assertIn(t, DATA_QUALITY_EXCLUDED_TICKERS, t)

    def test_existing_pool_helper_still_works(self):
        pool = pd.DataFrame({"ticker": [BAD, BAD2, "7203.T"], "x": [1, 2, 3]})
        self.assertEqual(list(exclude_data_quality_tickers(pool)["ticker"]), ["7203.T"])


class UniverseTest(unittest.TestCase):
    """ユニバース構築。`exclude` に渡せば `passes` が False になる。"""

    def _stats(self):
        return pd.DataFrame({
            "ticker": [BAD, BAD2, "7203.T"],
            "last_date": [pd.Timestamp("2026-09-16")] * 3,
            "bars": [2600, 2600, 2600],
            "last_close": [3000.0, 2000.0, 2500.0],
            "adv_jpy": [5e9, 5e9, 5e9], "adv_shares": [1e6] * 3,
            "turnover_last": [1e9] * 3, "ok_split": [True] * 3})

    def test_excluded_tickers_do_not_pass(self):
        f = apply_filters(self._stats(), 2e8, 200.0, 250,
                          exclude=sorted(DATA_QUALITY_EXCLUDED_TICKERS),
                          asof=pd.Timestamp("2026-09-17"))
        passed = set(f[f["passes"]]["ticker"])
        self.assertEqual(passed, {"7203.T"})

    def test_without_exclusion_they_would_pass(self):
        """**除外しなければ通ってしまう。** 出来高が正常なら ADV でも落ちない。"""
        f = apply_filters(self._stats(), 2e8, 200.0, 250, exclude=(),
                          asof=pd.Timestamp("2026-09-17"))
        self.assertEqual(set(f[f["passes"]]["ticker"]), {BAD, BAD2, "7203.T"})


class BenchmarkTest(unittest.TestCase):
    """**等加重なので、壊れた 1 銘柄がその日の平均を支配しうる。**"""

    def setUp(self):
        self.good = {f"{i}.T": frame() for i in range(1000, 1004)}

    def test_excluded_ticker_is_not_in_the_benchmark(self):
        t = frame().index[0]
        base = universe_benchmark_returns(self.good, t, (20,))[20]
        # 壊れた銘柄を足しても平均が動かない
        with_bad = dict(self.good)
        with_bad[BAD] = frame(close=1e9)
        out = universe_benchmark_returns(with_bad, t, (20,))[20]
        self.assertEqual(out["n_used"], base["n_used"])
        self.assertAlmostEqual(out["mean"], base["mean"])

    def test_a_non_excluded_ticker_does_move_it(self):
        """除外が効いていることの対照。**除外していない銘柄なら件数が増える。**"""
        with_other = dict(self.good)
        with_other["7203.T"] = frame()
        t = frame().index[0]
        out = universe_benchmark_returns(with_other, frame().index[0], (20,))[20]
        base = universe_benchmark_returns(self.good, t, (20,))[20]
        self.assertEqual(out["n_used"], base["n_used"] + 1)

    def test_explicit_excluded_argument_is_honoured(self):
        t = frame().index[0]
        out = universe_benchmark_returns(self.good, t, (20,), excluded={"1000.T"})[20]
        self.assertEqual(out["n_used"], 3)


class ReplayUniverseTest(unittest.TestCase):
    """バックテストのプール。ここを外せばベンチマークからも外れる。"""

    def test_excluded_ticker_is_not_in_the_pool(self):
        df = frame(n=400, volume=1e7)
        prepared = {t: {"df": df,
                        "adv": (df["Close"] * df["Volume"]).rolling(20, min_periods=20)
                        .mean().to_numpy(dtype=float)}
                    for t in (BAD, BAD2, "7203.T")}
        got = replay_mod.universe_at(prepared, {BAD, BAD2, "7203.T"},
                                     df.index[300], 2e8, 200.0, min_history_bars=250)
        self.assertEqual(got, ["7203.T"])


class FetchSanityTest(unittest.TestCase):
    """**置換する前の検査。** 1909.T の再取得で良い行を壊れた行にしてしまった反省。"""

    def test_market_cap_values_are_refused(self):
        issues = fetch_sanity_issues({BAD: frame(close=1.6e10), "7203.T": frame()})
        self.assertIn(BAD, issues)
        self.assertNotIn("7203.T", issues)
        self.assertIn("終値", issues[BAD])

    def test_all_zero_volume_is_refused(self):
        issues = fetch_sanity_issues({BAD: frame(volume=0.0)})
        self.assertIn(BAD, issues)
        self.assertIn("出来高", issues[BAD])

    def test_a_high_priced_but_sane_stock_passes(self):
        """**分割や急騰では引っかからない。** 見ているのは水準であって変化率ではない。"""
        self.assertEqual(fetch_sanity_issues({"9983.T": frame(close=FETCH_ABSURD_CLOSE / 2)}), {})

    def test_a_few_zero_volume_days_pass(self):
        df = frame()
        df.loc[df.index[:3], "Volume"] = 0.0
        self.assertEqual(fetch_sanity_issues({"7203.T": df}), {})

    def test_empty_input(self):
        self.assertEqual(fetch_sanity_issues({}), {})
        self.assertEqual(fetch_sanity_issues({"7203.T": pd.DataFrame()}), {})


if __name__ == "__main__":
    unittest.main()
