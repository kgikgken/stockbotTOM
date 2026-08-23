"""プール正規化と合成（DESIGN.md §6 / TASKS.md T-301）のテスト。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features import dimensions, pullback
from stockbot.scoring import composite
from stockbot.scoring.composite import (
    DIMENSIONS_BY_STATE,
    _band_score,
    _percentile_up,
    compute_composite_scores,
    feature_score,
)

# D8（採点しない）を除く全特徴量ID。行を作るときに欠損(NaN)で初期化しておく
SCORABLE_FEATURE_IDS = [fid for fid, dim, _d, _b in dimensions.FEATURE_METADATA if dim != "D8"]


def _row(ticker="1301.T", date="2026-08-21", state=pullback.STATE_BOUNCE, **overrides) -> dict:
    row = {"ticker": ticker, "date": pd.Timestamp(date), "state": state}
    for fid in SCORABLE_FEATURE_IDS:
        row[fid] = np.nan
    row.update(overrides)
    return row


class BandScoreBoundaryTest(unittest.TestCase):
    """必須テスト: 帯変換の境界値（帯内1、帯幅分外で0）。"""

    def test_inside_band_scores_one(self):
        self.assertEqual(_band_score(0.30, 0.10, 0.50), 1.0)
        self.assertEqual(_band_score(0.10, 0.10, 0.50), 1.0)
        self.assertEqual(_band_score(0.50, 0.10, 0.50), 1.0)

    def test_one_bandwidth_outside_scores_zero(self):
        lo, hi = 0.10, 0.50
        w = hi - lo
        self.assertAlmostEqual(_band_score(lo - w, lo, hi), 0.0)
        self.assertAlmostEqual(_band_score(hi + w, lo, hi), 0.0)

    def test_beyond_one_bandwidth_stays_clipped_at_zero(self):
        lo, hi = 0.10, 0.50
        w = hi - lo
        self.assertEqual(_band_score(lo - 5 * w, lo, hi), 0.0)
        self.assertEqual(_band_score(hi + 5 * w, lo, hi), 0.0)

    def test_half_bandwidth_outside_scores_half(self):
        lo, hi = 0.10, 0.50
        w = hi - lo
        self.assertAlmostEqual(_band_score(lo - w / 2, lo, hi), 0.5)

    def test_feature_score_band_uses_same_boundary(self):
        fid, lo, hi = "d3_position", 0.10, 0.50
        self.assertEqual(feature_score(fid, hi, np.array([])), 1.0)
        self.assertEqual(feature_score(fid, hi + (hi - lo), np.array([])), 0.0)


class MissingFeatureTest(unittest.TestCase):
    """必須テスト: 次元欠損時の扱い（欠損は0.5）。"""

    def test_nan_value_scores_half_regardless_of_direction(self):
        self.assertEqual(feature_score("d1_ma_stack", np.nan, np.array([1.0, 2.0])), 0.5)  # up
        self.assertEqual(feature_score("d3_maxdrop", np.nan, np.array([1.0, 2.0])), 0.5)  # down
        self.assertEqual(feature_score("d3_position", np.nan, np.array([])), 0.5)  # band
        self.assertEqual(feature_score("d3_bad_news", np.nan, np.array([])), 0.5)  # binary

    def test_dimension_score_falls_back_to_half_when_all_features_missing(self):
        pool = pd.DataFrame([_row()])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        for dim in composite.SCORED_DIMENSIONS:
            self.assertAlmostEqual(out.iloc[0][f"dim_{dim}_score"], 0.5, places=9)

    def test_empty_pool_percentile_is_nan_before_default_applied(self):
        self.assertTrue(np.isnan(_percentile_up(1.0, np.array([]))))
        # feature_score はこの NaN を最終的に 0.5 に丸める
        self.assertEqual(feature_score("d1_ma_stack", 1.0, np.array([])), 0.5)


class PercentileTest(unittest.TestCase):
    def test_percentile_counts_self_inclusive(self):
        pool_values = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(_percentile_up(20.0, pool_values), 0.5)
        self.assertAlmostEqual(_percentile_up(40.0, pool_values), 1.0)
        self.assertAlmostEqual(_percentile_up(5.0, pool_values), 0.0)

    def test_down_direction_inverts_percentile(self):
        pool_values = np.array([10.0, 20.0, 30.0, 40.0])
        up_pct = _percentile_up(20.0, pool_values)
        score = feature_score("d3_maxdrop", 20.0, pool_values)  # d3_maxdrop は down
        self.assertAlmostEqual(score, 1.0 - up_pct)


class BinaryFeatureTest(unittest.TestCase):
    def test_d3_bad_news_is_inverted(self):
        self.assertEqual(feature_score("d3_bad_news", True, np.array([])), 0.0)
        self.assertEqual(feature_score("d3_bad_news", False, np.array([])), 1.0)

    def test_d4_climax_is_excluded_from_dimension_score(self):
        self.assertNotIn("d4_climax", composite.SCORED_FEATURE_IDS_BY_DIM["D4"])
        self.assertTrue(np.isnan(feature_score("d4_climax", True, np.array([]))))


class DimensionSetByStateTest(unittest.TestCase):
    def test_forming_excludes_d6(self):
        self.assertNotIn("D6", DIMENSIONS_BY_STATE[pullback.STATE_FORMING])
        self.assertIn("D6", DIMENSIONS_BY_STATE[pullback.STATE_BOUNCE])
        self.assertIn("D6", DIMENSIONS_BY_STATE[pullback.STATE_BREAK])

    def test_composite_ignores_d6_for_forming_state(self):
        row_a = _row(state=pullback.STATE_FORMING, d6_trig_a=0.0)
        row_b = _row(state=pullback.STATE_FORMING, d6_trig_a=999.0)
        out_a = compute_composite_scores(pd.DataFrame([row_a]), pd.Timestamp("2026-08-21"),
                                         pool_days=1, log=lambda *a: None)
        out_b = compute_composite_scores(pd.DataFrame([row_b]), pd.Timestamp("2026-08-21"),
                                         pool_days=1, log=lambda *a: None)
        self.assertAlmostEqual(out_a.iloc[0]["score_v1"], out_b.iloc[0]["score_v1"], places=9)


class CompositeVariantsTest(unittest.TestCase):
    """受け入れ: 同じ入力で V1/V2/V3 が再現。"""

    def test_v1_v2_v3_reproducible_for_same_input(self):
        pool = pd.DataFrame([_row(d3_position=0.30)])
        out1 = compute_composite_scores(pool.copy(), pd.Timestamp("2026-08-21"),
                                        pool_days=1, log=lambda *a: None)
        out2 = compute_composite_scores(pool.copy(), pd.Timestamp("2026-08-21"),
                                        pool_days=1, log=lambda *a: None)
        for col in ("score_v1", "score_v2", "score_v3"):
            self.assertAlmostEqual(out1.iloc[0][col], out2.iloc[0][col], places=9)

    def test_v3_equals_v1_equal_weight(self):
        pool = pd.DataFrame([_row(d3_position=0.30)])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        self.assertAlmostEqual(out.iloc[0]["score_v1"], out.iloc[0]["score_v3"], places=9)

    def test_v2_weights_d3_double(self):
        row = _row(state=pullback.STATE_BOUNCE, d3_position=0.30)  # 帯内 -> dim_D3_score を押し上げる
        pool = pd.DataFrame([row])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        dims = DIMENSIONS_BY_STATE[pullback.STATE_BOUNCE]
        vals = [out.iloc[0][f"dim_{d}_score"] for d in dims]
        expected_v1 = float(np.mean(vals)) * 100
        weights = [2.0 if d == "D3" else 1.0 for d in dims]
        expected_v2 = float(np.average(vals, weights=weights)) * 100
        self.assertAlmostEqual(out.iloc[0]["score_v1"], expected_v1, places=6)
        self.assertAlmostEqual(out.iloc[0]["score_v2"], expected_v2, places=6)
        self.assertNotAlmostEqual(out.iloc[0]["score_v1"], out.iloc[0]["score_v2"], places=6)


class V3VolumeGateTest(unittest.TestCase):
    def test_gate_fails_when_pb_ratio_above_one(self):
        pool = pd.DataFrame([_row(d4_pb_ratio=1.5)])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        self.assertFalse(bool(out.iloc[0]["v3_volume_gate_pass"]))

    def test_gate_passes_when_pb_ratio_at_or_below_one(self):
        pool = pd.DataFrame([_row(d4_pb_ratio=1.0)])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        self.assertTrue(bool(out.iloc[0]["v3_volume_gate_pass"]))

    def test_gate_passes_when_missing(self):
        pool = pd.DataFrame([_row(d4_pb_ratio=np.nan)])
        out = compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=1, log=lambda *a: None)
        self.assertTrue(bool(out.iloc[0]["v3_volume_gate_pass"]))


class PoolDaysLogTest(unittest.TestCase):
    """受け入れ: プールの日数が足りない初期は利用可能な日数で計算しその旨をログ。"""

    def test_logs_when_pool_days_insufficient(self):
        pool = pd.DataFrame([_row(date="2026-08-21")])
        logs = []
        compute_composite_scores(pool, pd.Timestamp("2026-08-21"), pool_days=20, log=logs.append)
        self.assertTrue(any("プール日数不足" in m for m in logs))

    def test_no_log_when_pool_days_sufficient(self):
        rows = [_row(ticker="1301.T", date=str(d)) for d in pd.bdate_range("2026-07-24", periods=20)]
        pool = pd.DataFrame(rows)
        logs = []
        compute_composite_scores(pool, pd.Timestamp(pool.iloc[-1]["date"]), pool_days=20, log=logs.append)
        self.assertFalse(any("プール日数不足" in m for m in logs))


class LookaheadTest(unittest.TestCase):
    """必須テスト（DESIGN.md §11）: T 時点のスコアは T より後の日付の行に影響されない。"""

    def test_future_day_row_does_not_affect_todays_percentile_score(self):
        asof = pd.Timestamp("2026-08-21")
        today_row = _row(ticker="1301.T", date=asof, d1_ma_stack=0.5)
        past_row = _row(ticker="1671.T", date="2026-08-20", d1_ma_stack=0.4)
        future_row = _row(ticker="9999.T", date="2026-08-24", d1_ma_stack=100.0)  # 極端な未来値

        pool_without_future = pd.DataFrame([today_row, past_row])
        pool_with_future = pd.DataFrame([today_row, past_row, future_row])

        out_a = compute_composite_scores(pool_without_future, asof, pool_days=20, log=lambda *a: None)
        out_b = compute_composite_scores(pool_with_future, asof, pool_days=20, log=lambda *a: None)

        row_a = out_a[out_a["ticker"] == "1301.T"].iloc[0]
        row_b = out_b[out_b["ticker"] == "1301.T"].iloc[0]
        self.assertAlmostEqual(row_a["dim_D1_score"], row_b["dim_D1_score"], places=9)
        self.assertAlmostEqual(row_a["score_v1"], row_b["score_v1"], places=9)

    def test_future_row_is_excluded_from_pool_day_count(self):
        asof = pd.Timestamp("2026-08-21")
        rows = [_row(ticker="1301.T", date=asof), _row(ticker="1301.T", date="2026-08-24")]
        pool = pd.DataFrame(rows)
        logs = []
        compute_composite_scores(pool, asof, pool_days=20, log=logs.append)
        self.assertTrue(any("1/20" in m for m in logs))


if __name__ == "__main__":
    unittest.main()
