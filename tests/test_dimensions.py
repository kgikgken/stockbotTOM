"""特徴量 D1〜D8（T-204）。ID集合の一致・D6欠損・再計算一致・欠損数カウント。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.dimensions import D6_IDS, FEATURE_IDS, compute_dimensions
from stockbot.features.indicators import atr_wilder, sma
from stockbot.features.pullback import (
    STATE_BOUNCE,
    STATE_BREAK,
    STATE_COMPLETE,
    STATE_FORMING,
    STATE_INVALID,
    pullback_state,
)
from stockbot.features.swings import alternate_swings, detect_raw_swings

K = 3

# DESIGN.md §5 の表から独立に書き写した ID 集合（モジュール自身の一覧と突き合わせる）
EXPECTED_IDS = [
    "d1_ma_stack", "d1_slope200", "d1_slope75", "d1_maturity", "d1_leg_strength",
    "d2_rs60", "d2_rs120", "d2_rsline_pos",
    "d3_retrace", "d3_depth_atr", "d3_depth_pct", "d3_duration", "d3_maxdrop",
    "d3_slope", "d3_down_ratio", "d3_lower_highs", "d3_hl_dist", "d3_ma_dist",
    "d3_position", "d3_dev5", "d3_bad_news", "d3_template",
    "d4_pb_ratio", "d4_pb_slope", "d4_bounce_vol", "d4_climax",
    "d5_atr_ratio", "d5_range_contr", "d5_bbw_pct", "d5_step_ratio",
    "d6_trig_a", "d6_trig_b", "d6_break_r", "d6_bounce_str", "d6_wick", "d6_age",
    "d7_hl_streak", "d7_close_pos", "d7_weeks_above",
    "earnings_days", "exdiv_flag", "limit_flag", "regime", "breadth_75", "breadth_200",
]


def _build_acceptance_series():
    """T-203 と同じ合成 pullback 系列（形成中→反発開始→ブレイク→完了）にOHLCV/配当を付ける。"""
    warm_end = 180.0
    warmup = np.linspace(100.0, warm_end, 220)
    dip = np.array([180, 178, 176, 174, 176, 178, 180.0]) + (warmup[-1] - 180)
    rise = np.linspace(dip[-1], 220.0, 15)[1:]
    peak = np.array([220, 222, 224, 230, 224, 222, 220.0])
    close = np.concatenate([warmup, dip, rise, peak])
    pullback_forming = np.array([217, 214, 211, 209])
    bounce_trigger_day = np.array([214])
    post_bounce = np.array([217, 220])
    breakout = np.array([232])
    close_full = np.concatenate(
        [close, pullback_forming, bounce_trigger_day, post_bounce, breakout])

    idx = pd.bdate_range("2024-01-01", periods=len(close_full))
    close_s = pd.Series(close_full, dtype=float, index=idx)
    high_s = close_s + 0.3
    low_s = close_s - 0.3
    open_s = close_s - 0.1
    rng = np.random.default_rng(5)
    vol_s = pd.Series(rng.uniform(5e5, 2e6, len(close_full)), index=idx)
    div_s = pd.Series(0.0, index=idx)
    div_s.iloc[100] = 5.0  # 配当落ち日を1つ混ぜる（exdiv_flagの動作確認用、押し目区間の外）
    idx_close_s = pd.Series(500 + np.cumsum(rng.normal(0, 1, len(close_full))), index=idx)
    return open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s


def _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, t_pos, k=K):
    raw = detect_raw_swings(high_s, low_s, k)
    alt = alternate_swings(raw)
    sma5_s, sma200_s = sma(close_s, 5), sma(close_s, 200)
    atr14_s = atr_wilder(high_s, low_s, close_s, 14)
    pb = pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, t_pos, k)
    df, extra = compute_dimensions(open_s, high_s, low_s, close_s, vol_s, div_s, alt, pb,
                                   t_pos, k, idx_close=idx_close_s)
    return pb, df, extra


class FeatureIdSetTest(unittest.TestCase):
    def test_matches_design_md_table_exactly(self):
        self.assertEqual(set(FEATURE_IDS), set(EXPECTED_IDS))
        self.assertEqual(len(FEATURE_IDS), len(set(FEATURE_IDS)), "重複ID無し")

    def test_output_id_set_matches_metadata(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = _build_acceptance_series()
        _, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, 252)
        self.assertEqual(list(df["id"]), FEATURE_IDS)


class D6MissingInFormingTest(unittest.TestCase):
    def setUp(self):
        self.series = _build_acceptance_series()

    def test_d6_nan_when_forming(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = self.series
        pb, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, 248)
        self.assertEqual(pb["state"], STATE_FORMING)
        d6 = df[df["id"].isin(D6_IDS)]
        self.assertTrue(d6["value"].isna().all(), msg=d6)

    def test_d6_present_when_bounce_or_break(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = self.series
        for t, expected_state in ((252, STATE_BOUNCE), (253, STATE_BREAK)):
            pb, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, t)
            self.assertEqual(pb["state"], expected_state, msg=f"t={t}")
            d6 = df[df["id"].isin(D6_IDS)]
            self.assertTrue(d6["value"].notna().all(), msg=d6)

    def test_d6_nan_when_complete(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = self.series
        pb, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, 255)
        self.assertEqual(pb["state"], STATE_COMPLETE)
        d6 = df[df["id"].isin(D6_IDS)]
        self.assertTrue(d6["value"].isna().all(), msg=d6)


class MissingCountTest(unittest.TestCase):
    def test_no_structure_leaves_most_features_nan(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = _build_acceptance_series()
        # h0確定(247)より前は構造なし
        pb, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, 245)
        self.assertEqual(pb["state"], "no_structure")
        structure_dependent = {
            "d1_leg_strength", "d3_retrace", "d3_depth_atr", "d3_depth_pct", "d3_duration",
            "d3_maxdrop", "d3_slope", "d3_down_ratio", "d3_lower_highs", "d3_hl_dist",
            "d3_ma_dist", "d3_position", "d3_dev5", "d3_bad_news",
            "d4_pb_ratio", "d4_pb_slope", "d4_climax",
            "d5_range_contr", "d5_step_ratio",
            "d6_trig_a", "d6_trig_b", "d6_break_r", "d6_bounce_str", "d6_wick", "d6_age",
        }
        nan_ids = set(df[df["value"].isna()]["id"])
        self.assertTrue(structure_dependent.issubset(nan_ids))

    def test_missing_count_is_low_in_well_populated_state(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = _build_acceptance_series()
        _, df, _ = _compute(open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, 253)
        always_nan = {"d3_template", "regime", "breadth_75", "breadth_200"}
        nan_ids = set(df[df["value"].isna()]["id"])
        # 常時NaNの4つを除けば、ブレイク状態でここまで計算できていない特徴量はごく僅かのはず
        self.assertLessEqual(len(nan_ids - always_nan), 3, msg=nan_ids - always_nan)
        self.assertTrue(always_nan.issubset(nan_ids))


class RecalcConsistencyTest(unittest.TestCase):
    """DESIGN.md §11: T で切った系列 = 全期間で計算して T 行を取り出したもの（ローリング）。"""

    def test_truncated_matches_full_across_states_and_features(self):
        open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s = _build_acceptance_series()
        for t in (245, 247, 250, 252, 253, 255):
            _, full_df, full_extra = _compute(
                open_s, high_s, low_s, close_s, vol_s, div_s, idx_close_s, t)
            _, cut_df, cut_extra = _compute(
                open_s.iloc[:t + 1], high_s.iloc[:t + 1], low_s.iloc[:t + 1],
                close_s.iloc[:t + 1], vol_s.iloc[:t + 1], div_s.iloc[:t + 1],
                idx_close_s.iloc[:t + 1], t)
            for fid in FEATURE_IDS:
                full_v = full_df.loc[full_df["id"] == fid, "value"].iloc[0]
                cut_v = cut_df.loc[cut_df["id"] == fid, "value"].iloc[0]
                if isinstance(full_v, bool) or isinstance(cut_v, bool):
                    self.assertEqual(cut_v, full_v, msg=f"t={t} id={fid}")
                elif pd.isna(full_v):
                    self.assertTrue(pd.isna(cut_v), msg=f"t={t} id={fid}")
                else:
                    self.assertAlmostEqual(cut_v, full_v, places=9, msg=f"t={t} id={fid}")
            self.assertEqual(cut_extra, full_extra, msg=f"t={t}")


class SpotCheckTest(unittest.TestCase):
    """厄介な式のピンポイント手計算チェック。"""

    def test_d1_maturity_counts_from_most_recent_crossover(self):
        # 前半200本は緩やかに下落（SMA200が定義された時点でCloseはSMA200未満）、
        # その後上昇に転じてSMA200を上抜ける。手計算で t=208 が上抜け日と確認済み。
        n = 260
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = np.concatenate([np.linspace(105.0, 95.0, 200), np.linspace(95.0, 130.0, 60)])
        close_s = pd.Series(close, index=idx)
        high_s, low_s, open_s = close_s + 0.3, close_s - 0.3, close_s - 0.1
        vol_s = pd.Series(1e6, index=idx)
        div_s = pd.Series(0.0, index=idx)
        raw = detect_raw_swings(high_s, low_s, K)
        alt = alternate_swings(raw)
        sma5_s, sma200_s = sma(close_s, 5), sma(close_s, 200)
        atr14_s = atr_wilder(high_s, low_s, close_s, 14)
        t = n - 1
        pb = pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, t, K)
        df, _ = compute_dimensions(open_s, high_s, low_s, close_s, vol_s, div_s, alt, pb, t, K)
        maturity = df.loc[df["id"] == "d1_maturity", "value"].iloc[0]
        self.assertEqual(maturity, t - 208)

    def test_d5_bbw_pct_is_hundred_percent_for_widest_point(self):
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        rng = np.random.default_rng(2)
        # 後半にボラティリティを急拡大させ、直近が120日内で最もバンド幅が広くなるようにする
        close = 100 + np.cumsum(rng.normal(0, 0.3, n))
        close[-5:] = close[-6] + np.array([5, -8, 10, -12, 15])  # 急拡大
        close_s = pd.Series(close, index=idx)
        high_s, low_s, open_s = close_s + 0.3, close_s - 0.3, close_s - 0.1
        vol_s = pd.Series(1e6, index=idx)
        div_s = pd.Series(0.0, index=idx)
        raw = detect_raw_swings(high_s, low_s, K)
        alt = alternate_swings(raw)
        sma5_s, sma200_s = sma(close_s, 5), sma(close_s, 200)
        atr14_s = atr_wilder(high_s, low_s, close_s, 14)
        t = n - 1
        pb = pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, t, K)
        df, _ = compute_dimensions(open_s, high_s, low_s, close_s, vol_s, div_s, alt, pb, t, K)
        bbw_pct = df.loc[df["id"] == "d5_bbw_pct", "value"].iloc[0]
        self.assertAlmostEqual(bbw_pct, 1.0, places=9)

    def test_d3_bad_news_true_on_deep_single_day_drop(self):
        # H0から2.5ATR超の下落を1日で起こす
        warmup = np.linspace(154.0, 180.0, 220)
        dip = np.array([180, 178, 176, 174, 176, 178, 180.0])
        rise = np.linspace(dip[-1], 220.0, 15)[1:]
        peak = np.array([220, 222, 224, 230, 224, 222, 220.0])
        crash = np.array([190.0])  # H0直後、1日で大きく下落
        close_full = np.concatenate([warmup, dip, rise, peak, crash])
        idx = pd.bdate_range("2024-01-01", periods=len(close_full))
        close_s = pd.Series(close_full, index=idx)
        high_s, low_s, open_s = close_s + 0.3, close_s - 0.3, close_s - 0.1
        vol_s = pd.Series(1e6, index=idx)
        div_s = pd.Series(0.0, index=idx)
        raw = detect_raw_swings(high_s, low_s, K)
        alt = alternate_swings(raw)
        sma5_s, sma200_s = sma(close_s, 5), sma(close_s, 200)
        atr14_s = atr_wilder(high_s, low_s, close_s, 14)
        t = len(close_full) - 1
        pb = pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, t, K)
        self.assertIsNotNone(pb["h0"])
        df, _ = compute_dimensions(open_s, high_s, low_s, close_s, vol_s, div_s, alt, pb, t, K)
        bad_news = df.loc[df["id"] == "d3_bad_news", "value"].iloc[0]
        self.assertTrue(bool(bad_news))


if __name__ == "__main__":
    unittest.main()
