"""スイング検出（T-202）。手計算・交互化・DESIGN.md §11 の再計算一致テスト。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.swings import (
    SWING_HIGH,
    SWING_LOW,
    alternate_swings,
    detect_raw_swings,
    swings_as_of,
    swings_as_of_fresh,
)


def _series(values, name="High"):
    return pd.Series(values, dtype=float, name=name)


class DetectRawSwingsTest(unittest.TestCase):
    def test_matches_hand_calc_small_series(self):
        # k=1: 各インデックスは前後1本と比較するだけで判定できる
        high = _series([10, 12, 11, 9, 13, 12])
        low = _series([8, 9, 7, 6, 10, 9])
        raw = detect_raw_swings(high, low, k=1)
        highs = raw[raw["kind"] == SWING_HIGH]["index"].tolist()
        lows = raw[raw["kind"] == SWING_LOW]["index"].tolist()
        # i=1: 12>=10,12>11 → 高値。 i=3: 9<=11,9<13 かつ 9<=7? -> 左min(11,7)=7,9<=7 False → 対象外
        # i=4: 13>=max(9,11? ) 実際に手計算する
        self.assertIn(1, highs)
        self.assertIn(4, highs)
        self.assertIn(3, lows)

    def test_confirm_index_is_i_plus_k(self):
        high = _series([10, 12, 11, 9, 13, 12])
        low = _series([8, 9, 7, 6, 10, 9])
        raw = detect_raw_swings(high, low, k=1)
        for _, row in raw.iterrows():
            self.assertEqual(row["confirm_index"], row["index"] + 1)

    def test_left_ge_right_gt_asymmetry(self):
        # 左は同値可(>=)、右は厳密(>)。平坦区間での境界確認。
        high = _series([10, 10, 10, 9, 8])
        low = _series([1, 1, 1, 1, 1])  # 完全にフラットな安値は絶対にスイング安値にならない
        raw = detect_raw_swings(high, low, k=1)
        self.assertTrue((raw["kind"] == SWING_LOW).sum() == 0)
        # i=1: High[1]=10, left=[10] -> 10>=10 True。right=[10] -> 10>10 False → 高値ではない
        highs = raw[raw["kind"] == SWING_HIGH]["index"].tolist()
        self.assertNotIn(1, highs)


class AlternateSwingsTest(unittest.TestCase):
    def test_no_consecutive_same_kind_survives(self):
        rng = np.random.default_rng(3)
        n = 300
        close = 100 + np.cumsum(rng.normal(0, 1.5, n))
        high = pd.Series(close + rng.uniform(0.2, 1.5, n))
        low = pd.Series(close - rng.uniform(0.2, 1.5, n))
        raw = detect_raw_swings(high, low, k=3)
        alt = alternate_swings(raw)
        final = alt[alt["kept"] & alt["superseded_at"].isna()].sort_values("index")
        kinds = final["kind"].tolist()
        for a, b in zip(kinds, kinds[1:]):
            self.assertNotEqual(a, b, "交互化後に同種が連続してはいけない")

    def test_tie_keeps_the_earlier_one(self):
        # 同値の2つの高値が連続 → 先に来た方(index小さい方)が残る
        raw = pd.DataFrame([
            (2, SWING_HIGH, 100.0, 5),
            (10, SWING_HIGH, 100.0, 13),
        ], columns=["index", "kind", "value", "confirm_index"])
        alt = alternate_swings(raw)
        final = alt[alt["kept"] & alt["superseded_at"].isna()]
        self.assertEqual(list(final["index"]), [2])


class ReplacementAcrossConfirmLagTest(unittest.TestCase):
    """質問ログ 2026-08-22 T-202 で確認した「末尾のみ置換される」ケースの直接検証。

    k=2。i1=2(高値=12,confirm=4)、i2=12(高値=14,confirm=14)。安値は完全フラットな
    ので一切検出されない(左右対称チェックの右側が厳密不等号のため)。i2 > i1+k な
    ので、DESIGN.md 質問ログの証明どおり i2 のみが i1 を置換しうる。
    """

    def setUp(self):
        self.k = 2
        # 17本: 0..14 が本編、15,16 は「i2 確定後も未来データが漏れないこと」の確認用の余白
        self.high = _series([10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10])
        self.low = _series([1.0] * len(self.high))
        self.i1, self.i2 = 2, 12

    def test_raw_detection_exactly_two_highs_no_lows(self):
        raw = detect_raw_swings(self.high, self.low, self.k)
        self.assertEqual(set(raw[raw["kind"] == SWING_HIGH]["index"]), {self.i1, self.i2})
        self.assertEqual((raw["kind"] == SWING_LOW).sum(), 0)

    def test_alternation_replaces_only_at_confirm_of_i2(self):
        raw = detect_raw_swings(self.high, self.low, self.k)
        alt = alternate_swings(raw)

        row_i1 = alt[alt["index"] == self.i1].iloc[0]
        row_i2 = alt[alt["index"] == self.i2].iloc[0]
        self.assertTrue(row_i1["kept"])
        self.assertEqual(row_i1["superseded_at"], self.i2 + self.k)  # =14
        self.assertTrue(row_i2["kept"])
        self.assertTrue(pd.isna(row_i2["superseded_at"]))

        # T=13 まではi1のみが「今分かっている」交互化結果
        self.assertEqual(list(swings_as_of(alt, 13)["index"]), [self.i1])
        # T=14 でi2が確定し、その瞬間にi1を置換する
        self.assertEqual(list(swings_as_of(alt, 14)["index"]), [self.i2])

    def test_truncated_recompute_matches_full_table_at_every_t(self):
        """DESIGN.md §11: 生検出は Tで切った=全期間をi<=T-kで絞ったもの。
        交互化は Tで切った系列で生検出→交互化 = 全期間テーブルをTで問い合わせた結果。
        """
        alt_full = alternate_swings(detect_raw_swings(self.high, self.low, self.k))
        for t in (1, 3, 4, 5, 10, 13, 14, 15, 16):
            truncated_high = self.high.iloc[: t + 1]
            truncated_low = self.low.iloc[: t + 1]
            fresh = swings_as_of_fresh(truncated_high, truncated_low, self.k, t)
            from_full = swings_as_of(alt_full, t)
            self.assertListEqual(list(fresh["index"]), list(from_full["index"]),
                                 msg=f"t={t}")
            self.assertListEqual(list(fresh["kind"]), list(from_full["kind"]), msg=f"t={t}")

    def test_incremental_update_from_t_minus_1_matches_recompute_at_t(self):
        """T−1時点の交互化結果に、Tで新たに確定した生スイングを逐次適用した結果 == T時点の結果。

        独立実装（このテスト内）でインクリメンタル更新を行い、モジュールの
        バッチ計算結果と突き合わせる。
        """
        raw = detect_raw_swings(self.high, self.low, self.k)
        alt_full = alternate_swings(raw)

        def apply_new_swings(prev_tail, new_rows):
            """prev_tail: (kind, value) or None。new_rows: 昇順の (kind, value) のリスト。"""
            tail = prev_tail
            for kind, value in new_rows:
                if tail is not None and tail[0] == kind:
                    more_extreme = value > tail[1] if kind == SWING_HIGH else value < tail[1]
                    if more_extreme:
                        tail = (kind, value)
                    # else: 捨てる(tail 不変)
                else:
                    tail = (kind, value)
            return tail

        prev_tail = None
        for t in range(0, 17):
            new_rows = raw[raw["confirm_index"] == t][["kind", "value"]].sort_values(
                ["kind"]).itertuples(index=False, name=None)
            prev_tail = apply_new_swings(prev_tail, list(new_rows))
            got = swings_as_of(alt_full, t)
            if prev_tail is None:
                self.assertEqual(len(got), 0, msg=f"t={t}")
            else:
                self.assertEqual(len(got), 1, msg=f"t={t}")
                self.assertEqual(got.iloc[0]["kind"], prev_tail[0], msg=f"t={t}")
                self.assertAlmostEqual(got.iloc[0]["value"], prev_tail[1], msg=f"t={t}")


class RecalcConsistencyRandomSeriesTest(unittest.TestCase):
    """合成系列（乱数）でも §11 の一致が保たれることを広く確認する。"""

    def setUp(self):
        rng = np.random.default_rng(11)
        n = 200
        close = 500 + np.cumsum(rng.normal(0, 3, n))
        self.high = pd.Series(close + rng.uniform(0.5, 3, n))
        self.low = pd.Series(close - rng.uniform(0.5, 3, n))
        self.k = 3
        self.alt_full = alternate_swings(detect_raw_swings(self.high, self.low, self.k))

    def test_raw_detection_recalc_consistency(self):
        full_raw = detect_raw_swings(self.high, self.low, self.k)
        for t in (20, 50, 100, 150, 199):
            cut_raw = detect_raw_swings(self.high.iloc[:t + 1], self.low.iloc[:t + 1], self.k)
            expected = full_raw[full_raw["index"] <= t - self.k]
            pd.testing.assert_frame_equal(
                cut_raw.reset_index(drop=True), expected.reset_index(drop=True))
            self.assertTrue((cut_raw["index"] <= t - self.k).all())

    def test_alternation_recalc_consistency(self):
        for t in (20, 50, 100, 150, 199):
            fresh = swings_as_of_fresh(self.high.iloc[:t + 1], self.low.iloc[:t + 1], self.k, t)
            from_full = swings_as_of(self.alt_full, t)
            self.assertListEqual(list(fresh["index"]), list(from_full["index"]), msg=f"t={t}")


if __name__ == "__main__":
    unittest.main()
