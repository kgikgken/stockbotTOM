"""ラベル（DESIGN.md §9 / TASKS.md T-401）のテスト。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.validation.labels import (
    LABEL_FAILURE,
    LABEL_SUCCESS,
    LABEL_UNDETERMINED,
    compute_labels,
    excess_return,
    mfe_mae,
    outcome_label,
    universe_benchmark_returns,
)

IDX = pd.bdate_range("2024-01-01", periods=25)


def _series(values: list[float]) -> pd.Series:
    padded = values + [np.nan] * (len(IDX) - len(values))
    return pd.Series(padded[: len(IDX)], index=IDX)


class HandCalcTest(unittest.TestCase):
    """受け入れ: 手計算の小例と一致。"""

    T_POS = 10
    H0_HIGH = 110.0
    LP_VALUE = 95.0
    ATR_T = 5.0
    N = 5

    def _build(self):
        # T_POS=10 (open[11]=Open[T+1]=entry=100)。位置10はT自身のプレースホルダーで未使用。
        # T+3日目(位置13)にHigh>110かつLow<95が同時発生
        open_ = [0.0] * self.T_POS + [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        high = [0.0] * self.T_POS + [0.0, 105.0, 108.0, 111.0, 115.0, 107.0]
        low = [0.0] * self.T_POS + [0.0, 98.0, 96.0, 94.0, 90.0, 99.0]
        close = [0.0] * self.T_POS + [0.0, 102.0, 104.0, 109.0, 112.0, 103.0]
        return _series(open_), _series(high), _series(low), _series(close)

    def test_hand_calc_label_mfe_mae_and_excess_return(self):
        open_, high, low, close = self._build()
        benchmarks = {3: 0.02, 5: 0.01, 10: np.nan, 15: np.nan, 20: np.nan, 30: np.nan}
        out = compute_labels(close, open_, high, low, self.T_POS, self.H0_HIGH,
                             self.LP_VALUE, self.ATR_T, benchmarks, n=self.N)

        # 成功/失敗/未決: T+3日目(位置13)でHigh=111>110 かつ Low=94<95 が同時発生 → 失敗
        self.assertEqual(out["label"], LABEL_FAILURE)
        self.assertEqual(out["hit_day"], 3)

        # MFE/MAE: 判定日(3日目)に関わらずT+1..T+5固定窓。max(High)=115(4日目)、min(Low)=90(4日目)
        self.assertAlmostEqual(out["mfe"], (115.0 - 100.0) / 5.0, places=9)
        self.assertAlmostEqual(out["mae"], (100.0 - 90.0) / 5.0, places=9)

        # 超過リターン: r_h = ln(Close[T+h]/Open[T+1]) - benchmark
        self.assertAlmostEqual(out["r_3"], np.log(109.0 / 100.0) - 0.02, places=9)
        self.assertAlmostEqual(out["r_5"], np.log(103.0 / 100.0) - 0.01, places=9)
        # h=10/15/20/30 はデータ不足のためNaN
        for hz in (10, 15, 20, 30):
            self.assertTrue(np.isnan(out[f"r_{hz}"]), msg=hz)

        # censored_at: T+1以降、系列には25-11=14本あるが N=5 で頭打ちなので5
        self.assertEqual(out["censored_at"], 5)


class OutcomeLabelPriorityTest(unittest.TestCase):
    """成功/失敗/未決の判定順（同日に両方成立したら失敗を優先）。"""

    def test_success_only(self):
        high = np.array([0, 0, 105.0, 106.0])
        low = np.array([0, 0, 99.0, 98.0])
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=5)
        self.assertEqual(out["label"], LABEL_SUCCESS)
        self.assertEqual(out["hit_day"], 1)

    def test_failure_only(self):
        high = np.array([0, 0, 100.0, 106.0])
        low = np.array([0, 0, 85.0, 98.0])
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=5)
        self.assertEqual(out["label"], LABEL_FAILURE)
        self.assertEqual(out["hit_day"], 1)

    def test_same_day_both_is_failure(self):
        high = np.array([0, 0, 105.0])  # 成功条件も満たすが
        low = np.array([0, 0, 85.0])   # 失敗条件も同日に満たす
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=5)
        self.assertEqual(out["label"], LABEL_FAILURE)
        self.assertEqual(out["hit_day"], 1)

    def test_undetermined_when_neither_triggers_within_n(self):
        high = np.array([0, 0, 101.0, 102.0, 103.0])
        low = np.array([0, 0, 95.0, 94.0, 93.0])
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=3)
        self.assertEqual(out["label"], LABEL_UNDETERMINED)
        self.assertIsNone(out["hit_day"])

    def test_undetermined_when_data_runs_out_before_n(self):
        # t_pos=1, n=10だが系列はt_pos+3までしか無い。その範囲で何も起きなければ未決
        high = np.array([0, 0, 101.0, 102.0])
        low = np.array([0, 0, 95.0, 94.0])
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=10)
        self.assertEqual(out["label"], LABEL_UNDETERMINED)

    def test_first_triggering_day_wins_not_a_later_one(self):
        # 2日目に成功、3日目に失敗。2日目が先に成立するので成功が確定
        high = np.array([0, 0, 105.0, 106.0])
        low = np.array([0, 0, 99.0, 80.0])
        out = outcome_label(high, low, t_pos=1, h0_high=104.0, lp_value=90.0, n=5)
        self.assertEqual(out["label"], LABEL_SUCCESS)
        self.assertEqual(out["hit_day"], 1)


class MfeMaeTest(unittest.TestCase):
    def test_insufficient_future_data_returns_nan(self):
        high = np.array([100.0])
        low = np.array([90.0])
        open_ = np.array([95.0])
        out = mfe_mae(high, low, open_, t_pos=0, n=5, atr_t=5.0)
        self.assertTrue(np.isnan(out["mfe"]))
        self.assertTrue(np.isnan(out["mae"]))

    def test_non_positive_atr_returns_nan(self):
        high = np.array([0, 105.0, 108.0])
        low = np.array([0, 98.0, 96.0])
        open_ = np.array([0, 100.0, 101.0])
        for bad_atr in (0.0, -1.0, np.nan):
            out = mfe_mae(high, low, open_, t_pos=0, n=5, atr_t=bad_atr)
            self.assertTrue(np.isnan(out["mfe"]), msg=bad_atr)
            self.assertTrue(np.isnan(out["mae"]), msg=bad_atr)


class ExcessReturnTest(unittest.TestCase):
    def test_matches_hand_calc(self):
        close = np.array([0.0, 0.0, 0.0, 109.0])
        open_ = np.array([0.0, 100.0, 0.0, 0.0])
        r = excess_return(close, open_, t_pos=0, h=3, benchmark=0.02)
        self.assertAlmostEqual(r, np.log(109.0 / 100.0) - 0.02, places=9)

    def test_insufficient_future_data_returns_nan(self):
        close = np.array([0.0, 0.0])
        open_ = np.array([0.0, 100.0])
        r = excess_return(close, open_, t_pos=0, h=3, benchmark=0.0)
        self.assertTrue(np.isnan(r))

    def test_nan_benchmark_returns_nan(self):
        close = np.array([0.0, 0.0, 0.0, 109.0])
        open_ = np.array([0.0, 100.0, 0.0, 0.0])
        r = excess_return(close, open_, t_pos=0, h=3, benchmark=np.nan)
        self.assertTrue(np.isnan(r))


class UniverseBenchmarkReturnsTest(unittest.TestCase):
    """T-401 条件付き承認: 生存バイアス対策（打ち切り扱い）と n_used/n_truncated/n_excluded。"""

    def _df(self, opens, closes):
        idx = pd.bdate_range("2024-01-01", periods=len(opens))
        return pd.DataFrame({"Open": opens, "Close": closes}, index=idx)

    def test_simple_average_across_universe(self):
        date_t = pd.bdate_range("2024-01-01", periods=1)[0]  # 位置0がT
        # T+1=位置1の始値、T+3=位置3の終値。どちらも打ち切り無し
        a = self._df([0, 100.0, 0, 0], [0, 0, 0, 110.0])
        b = self._df([0, 100.0, 0, 0], [0, 0, 0, 121.0])
        universe = {"A": a, "B": b}
        result = universe_benchmark_returns(universe, date_t, h_list=(3,))
        expected = float(np.mean([np.log(110.0 / 100.0), np.log(121.0 / 100.0)]))
        self.assertAlmostEqual(result[3]["mean"], expected, places=9)
        self.assertEqual(result[3]["n_used"], 2)
        self.assertEqual(result[3]["n_truncated"], 0)
        self.assertEqual(result[3]["n_excluded"], 0)

    def test_ticker_without_date_t_is_excluded(self):
        date_t = pd.Timestamp("2024-01-01")
        a = self._df([0, 100.0, 0, 0], [0, 0, 0, 110.0])
        b = pd.DataFrame({"Open": [1.0], "Close": [1.0]},
                         index=pd.bdate_range("2030-01-01", periods=1))  # date_t が無い
        result = universe_benchmark_returns({"A": a, "B": b}, date_t, h_list=(3,))
        self.assertAlmostEqual(result[3]["mean"], np.log(110.0 / 100.0), places=9)
        self.assertEqual(result[3]["n_used"], 1)
        self.assertEqual(result[3]["n_excluded"], 1)

    def test_missing_th_close_is_truncated_not_excluded(self):
        """生存バイアス対策の核心: T+h が無い銘柄は除外せず、最後の有効な終値で打ち切る。"""
        date_t = pd.bdate_range("2024-01-01", periods=1)[0]
        a = self._df([0, 100.0, 0, 0], [0, 0, 0, 110.0])  # T+3まで揃っている
        # B は T, T+1, T+2 の3本しか無い（上場廃止等でT+3が存在しない）
        idx_b = pd.bdate_range("2024-01-01", periods=3)
        b = pd.DataFrame({"Open": [0.0, 100.0, 0.0], "Close": [0.0, 0.0, 90.0]}, index=idx_b)
        result = universe_benchmark_returns({"A": a, "B": b}, date_t, h_list=(3,))
        # Bは T+2 の終値(90)までの打ち切りリターン ln(90/100) を使う（除外しない）
        expected = float(np.mean([np.log(110.0 / 100.0), np.log(90.0 / 100.0)]))
        self.assertAlmostEqual(result[3]["mean"], expected, places=9)
        self.assertEqual(result[3]["n_used"], 2)
        self.assertEqual(result[3]["n_truncated"], 1)
        self.assertEqual(result[3]["n_excluded"], 0)

    def test_ticker_without_t1_open_is_excluded(self):
        date_t = pd.bdate_range("2024-01-01", periods=1)[0]
        a = self._df([0, 100.0, 0, 0], [0, 0, 0, 110.0])
        # B は T しか無い（T+1 の始値自体が存在しない）→ 打ち切りようがなく除外
        idx_b = pd.bdate_range("2024-01-01", periods=1)
        b = pd.DataFrame({"Open": [1.0], "Close": [1.0]}, index=idx_b)
        result = universe_benchmark_returns({"A": a, "B": b}, date_t, h_list=(3,))
        self.assertAlmostEqual(result[3]["mean"], np.log(110.0 / 100.0), places=9)
        self.assertEqual(result[3]["n_used"], 1)
        self.assertEqual(result[3]["n_excluded"], 1)

    def test_no_qualifying_ticker_gives_nan(self):
        date_t = pd.bdate_range("2024-01-01", periods=1)[0]
        idx_b = pd.bdate_range("2024-01-01", periods=1)
        b = pd.DataFrame({"Open": [1.0], "Close": [1.0]}, index=idx_b)  # T+1が無い
        result = universe_benchmark_returns({"B": b}, date_t, h_list=(3,))
        self.assertTrue(np.isnan(result[3]["mean"]))
        self.assertEqual(result[3]["n_used"], 0)
        self.assertEqual(result[3]["n_excluded"], 1)


class CensoredAtTest(unittest.TestCase):
    """T-401 条件付き承認: censored_at で「上場廃止等でデータが尽きた」ことを区別できる。"""

    def test_censored_at_is_capped_at_n_when_enough_data(self):
        open_, high, low, close = HandCalcTest()._build()
        benchmarks = {h: np.nan for h in (3, 5, 10, 15, 20, 30)}
        out = compute_labels(close, open_, high, low, HandCalcTest.T_POS, HandCalcTest.H0_HIGH,
                             HandCalcTest.LP_VALUE, HandCalcTest.ATR_T, benchmarks, n=5)
        self.assertEqual(out["censored_at"], 5)

    @staticmethod
    def _unpadded_series(values: list[float]) -> pd.Series:
        # _series() は IDX(25本)に合わせてNaNで埋めてしまうため、系列が本当に短い
        # （上場廃止で打ち切られた）ケースを再現するにはそのままの長さで作る
        return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))

    def test_censored_at_reflects_series_ending_early(self):
        # T_POS=10、系列はT+1,T+2の2本しか無い（n=15を要求してもcensored_atは2）
        t_pos = 10
        open_ = self._unpadded_series([0.0] * t_pos + [0.0, 100.0, 101.0])
        high = self._unpadded_series([0.0] * t_pos + [0.0, 105.0, 106.0])
        low = self._unpadded_series([0.0] * t_pos + [0.0, 98.0, 99.0])
        close = self._unpadded_series([0.0] * t_pos + [0.0, 102.0, 103.0])
        benchmarks = {h: np.nan for h in (3, 5, 10, 15, 20, 30)}
        out = compute_labels(close, open_, high, low, t_pos, h0_high=200.0, lp_value=50.0,
                             atr_t=5.0, universe_benchmarks=benchmarks, n=15)
        self.assertEqual(out["censored_at"], 2)

    def test_censored_at_is_zero_when_t_is_the_last_bar(self):
        t_pos = 10
        open_ = self._unpadded_series([0.0] * (t_pos + 1))
        high = self._unpadded_series([0.0] * (t_pos + 1))
        low = self._unpadded_series([0.0] * (t_pos + 1))
        close = self._unpadded_series([0.0] * (t_pos + 1))
        benchmarks = {h: np.nan for h in (3, 5, 10, 15, 20, 30)}
        out = compute_labels(close, open_, high, low, t_pos, h0_high=200.0, lp_value=50.0,
                             atr_t=5.0, universe_benchmarks=benchmarks, n=15)
        self.assertEqual(out["censored_at"], 0)
        self.assertEqual(out["label"], LABEL_UNDETERMINED)


class NoLookaheadTest(unittest.TestCase):
    """必須テスト: T 以前の行を変えてもラベルが変わらない。"""

    T_POS = 10
    H0_HIGH = 110.0
    LP_VALUE = 95.0
    ATR_T = 5.0
    N = 5

    def _series_with_prefix(self, prefix_seed: float):
        rng = np.random.default_rng(int(prefix_seed))
        prefix_open = list(rng.uniform(50, 150, size=self.T_POS))
        prefix_high = list(rng.uniform(50, 150, size=self.T_POS))
        prefix_low = list(rng.uniform(50, 150, size=self.T_POS))
        prefix_close = list(rng.uniform(50, 150, size=self.T_POS))
        tail_open = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        tail_high = [0.0, 105.0, 108.0, 111.0, 115.0, 107.0]
        tail_low = [0.0, 98.0, 96.0, 94.0, 90.0, 99.0]
        tail_close = [0.0, 102.0, 104.0, 109.0, 112.0, 103.0]
        return (_series(prefix_open + tail_open), _series(prefix_high + tail_high),
                _series(prefix_low + tail_low), _series(prefix_close + tail_close))

    def test_changing_bars_before_t_does_not_change_the_label(self):
        benchmarks = {3: 0.02, 5: 0.01, 10: np.nan, 15: np.nan, 20: np.nan, 30: np.nan}
        results = []
        for seed in (1, 2, 3):
            open_, high, low, close = self._series_with_prefix(seed)
            out = compute_labels(close, open_, high, low, self.T_POS, self.H0_HIGH,
                                 self.LP_VALUE, self.ATR_T, benchmarks, n=self.N)
            results.append(out)
        first = results[0]
        for other in results[1:]:
            for key in first:
                if isinstance(first[key], float) and np.isnan(first[key]):
                    self.assertTrue(np.isnan(other[key]), msg=key)
                else:
                    self.assertEqual(other[key], first[key], msg=key)

    def test_universe_benchmark_unaffected_by_data_before_t(self):
        date_t = pd.bdate_range("2024-01-01", periods=1)[0]
        idx = pd.bdate_range("2024-01-01", periods=4)
        # T(位置0)自身のOpenだけを変える。T+1の始値・T+3の終値は同じ
        a1 = pd.DataFrame({"Open": [1.0, 100.0, 1.0, 0.0], "Close": [0, 0, 0, 110.0]}, index=idx)
        a2 = pd.DataFrame({"Open": [999.0, 100.0, 1.0, 0.0], "Close": [0, 0, 0, 110.0]}, index=idx)
        r1 = universe_benchmark_returns({"A": a1}, date_t, h_list=(3,))
        r2 = universe_benchmark_returns({"A": a2}, date_t, h_list=(3,))
        self.assertEqual(r1[3]["mean"], r2[3]["mean"])


if __name__ == "__main__":
    unittest.main()
