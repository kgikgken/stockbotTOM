"""押し目構造と状態（T-203）。合成系列での状態遷移・失効条件の独立性・再計算一致テスト。"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.indicators import atr_wilder, sma
from stockbot.features.pullback import (
    STATE_BOUNCE,
    STATE_BREAK,
    STATE_COMPLETE,
    STATE_FORMING,
    STATE_INVALID,
    STATE_NO_STRUCTURE,
    find_h0,
    pullback_state,
)
from stockbot.features.swings import SWING_HIGH, SWING_LOW, alternate_swings, detect_raw_swings

K = 3


def _series(close_full):
    idx = pd.bdate_range("2024-01-01", periods=len(close_full))
    close_s = pd.Series(close_full, dtype=float, index=idx)
    high_s = close_s + 0.3
    low_s = close_s - 0.3
    return high_s, low_s, close_s


def build_leg(l0_value, h0_value, tail, n_warmup=220, warm_start=None):
    """warmup（単調増加、スイングなし）→ L0のV字ディップ → 上昇 → H0のピーク → tail。

    warmup は L0 の少し下から緩やかに立ち上がる直線なので、区間内に余計なフラクタル
    スイングを作らない（単調増加区間は fractal 条件を満たさない）。
    戻り値: (Close 配列, h0 の絶対位置)。
    """
    if warm_start is None:
        warm_start = l0_value - 20
    warm_end = l0_value + 6
    warmup = np.linspace(warm_start, warm_end, n_warmup)
    half = warm_end - l0_value
    dip = np.array([warm_end, warm_end - half * 2 / 3, warm_end - half, l0_value,
                    warm_end - half, warm_end - half * 2 / 3, warm_end], dtype=float)
    rise = np.linspace(dip[-1], h0_value - 10, 15)[1:]
    peak = np.array([h0_value - 10, h0_value - 8, h0_value - 6, h0_value,
                     h0_value - 6, h0_value - 8, h0_value - 10], dtype=float)
    close = np.concatenate([warmup, dip, rise, peak])
    h0_pos = len(warmup) + len(dip) + len(rise) + 3
    close_full = np.concatenate([close, np.asarray(tail, dtype=float)])
    return close_full, h0_pos


def compute(high_s, low_s, close_s, t_pos, k=K, **kwargs):
    raw = detect_raw_swings(high_s, low_s, k)
    alt = alternate_swings(raw)
    sma5_s = sma(close_s, 5)
    sma200_s = sma(close_s, 200)
    atr14_s = atr_wilder(high_s, low_s, close_s, 14)
    return pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, t_pos, k, **kwargs)


class AcceptanceTransitionTest(unittest.TestCase):
    """受け入れ条件: 合成 pullback 系列で 形成中 → 反発開始 → 完了 の順に状態が遷移する。"""

    @classmethod
    def setUpClass(cls):
        warm_end = 180.0
        warmup = np.linspace(100.0, warm_end, 220)
        dip = np.array([180, 178, 176, 174, 176, 178, 180.0]) + (warmup[-1] - 180)
        rise = np.linspace(dip[-1], 220.0, 15)[1:]
        peak = np.array([220, 222, 224, 230, 224, 222, 220.0])
        close = np.concatenate([warmup, dip, rise, peak])
        pullback_forming = np.array([217, 214, 211, 209])   # d=1..4、押し目形成中
        bounce_trigger_day = np.array([214])                # 反発トリガー発火日
        post_bounce = np.array([217, 220])
        breakout = np.array([232])                          # H0.high(230.3) を上抜け
        close_full = np.concatenate(
            [close, pullback_forming, bounce_trigger_day, post_bounce, breakout])
        cls.high_s, cls.low_s, cls.close_s = _series(close_full)
        cls.h0 = 244  # peak 配列の中央（0-indexed）

    def test_no_structure_before_h0_confirms(self):
        for t in (self.h0, self.h0 + 1, self.h0 + 2):  # confirm_index = h0+k = 247
            r = compute(self.high_s, self.low_s, self.close_s, t)
            self.assertEqual(r["state"], STATE_NO_STRUCTURE, msg=f"t={t}")

    def test_state_sequence_forming_then_bounce_then_complete(self):
        states = {}
        for t in range(247, len(self.close_s)):
            r = compute(self.high_s, self.low_s, self.close_s, t)
            states[t] = r["state"]
        seq = [states[t] for t in sorted(states)]
        # 形成中が最初に現れ、その後に反発開始、その後に完了が現れる（この相対順序）
        i_forming = seq.index(STATE_FORMING)
        i_bounce = seq.index(STATE_BOUNCE)
        i_complete = seq.index(STATE_COMPLETE)
        self.assertLess(i_forming, i_bounce)
        self.assertLess(i_bounce, i_complete)
        self.assertEqual(seq[-1], STATE_COMPLETE)

    def test_exactly_one_state_per_day(self):
        valid = {STATE_COMPLETE, STATE_INVALID, STATE_BREAK, STATE_BOUNCE,
                STATE_FORMING, STATE_NO_STRUCTURE}
        for t in range(len(self.close_s)):
            r = compute(self.high_s, self.low_s, self.close_s, t)
            self.assertIn(r["state"], valid, msg=f"t={t}")
            self.assertEqual(len({r["state"]}), 1)

    def test_recalc_consistency_truncated_matches_full(self):
        for t in range(247, len(self.close_s)):
            full = compute(self.high_s, self.low_s, self.close_s, t)
            cut = compute(self.high_s.iloc[:t + 1], self.low_s.iloc[:t + 1],
                          self.close_s.iloc[:t + 1], t)
            self.assertEqual(cut["state"], full["state"], msg=f"t={t}")
            self.assertEqual(cut["h0"], full["h0"], msg=f"t={t}")
            self.assertEqual(cut["l0"], full["l0"], msg=f"t={t}")
            self.assertEqual(cut["lp"], full["lp"], msg=f"t={t}")
            if not pd.isna(full["depth_pct"]):
                self.assertAlmostEqual(cut["depth_pct"], full["depth_pct"], places=9, msg=f"t={t}")


class InvalidationConditionsTest(unittest.TestCase):
    """§4.2 の失効条件が、それぞれ単独で（他を満たしたまま）状態を失効にすること。"""

    def _assert_others_hold(self, r, except_field):
        checks = {
            "d": 2 <= r["d"] <= 25,
            "lp_gt_l0": r["lp_value"] > r["l0_low"],
            "depth": r["depth_pct"] <= 0.25,
        }
        for name, ok in checks.items():
            if name == except_field:
                continue
            self.assertTrue(ok, msg=f"{name} should still hold: {r}")

    def test_d_greater_than_25_alone(self):
        tail = np.tile([218.0, 220.0, 222.0, 220.0], 8)[:30]  # d=30 > 25、他は健全
        close_full, h0 = build_leg(200.0, 230.0, tail, warm_start=195.0)
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1
        r = compute(high_s, low_s, close_s, T)
        self.assertEqual(r["state"], STATE_INVALID)
        self.assertGreater(r["d"], 25)
        self._assert_others_hold(r, "d")

    def test_lp_breaches_l0_low_alone(self):
        # L0.low=173.7、25%押し目の下限は172.725。173.4はその間 -> depth_pct はまだ健全
        tail = np.array([220.0, 210.0, 200.0, 190.0, 180.0, 173.4])
        close_full, h0 = build_leg(174.0, 230.0, tail, warm_start=154.0)
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1
        r = compute(high_s, low_s, close_s, T)
        self.assertEqual(r["state"], STATE_INVALID)
        self.assertLessEqual(r["lp_value"], r["l0_low"])
        self._assert_others_hold(r, "lp_gt_l0")

    def test_depth_pct_over_25_percent_alone(self):
        # L0.low=149.7 と十分深いので、depth_pct が先に閾値を超えても Lp>L0.low は健在
        tail = np.array([214, 208, 202, 196, 190, 184, 178, 172, 168, 165.0])
        close_full, h0 = build_leg(150.0, 230.0, tail)
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1
        r = compute(high_s, low_s, close_s, T)
        self.assertEqual(r["state"], STATE_INVALID)
        self.assertGreater(r["depth_pct"], 0.25)
        self._assert_others_hold(r, "depth")
        self.assertTrue(r["is_deep"])  # depth_pct(0.25超) > deep_depth(0.10)
        self.assertFalse(r["is_shallow"])

    def test_close_below_sma200_alone(self):
        tail = np.array([215.0, 210.0, 205.0, 202.0])
        close_full, h0 = build_leg(200.0, 230.0, tail, warm_start=195.0)
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1
        sma200_t = sma(close_s, 200).iloc[T]
        r = compute(high_s, low_s, close_s, T)
        self.assertEqual(r["state"], STATE_INVALID)
        self.assertLess(float(close_s.iloc[T]), sma200_t)
        self._assert_others_hold(r, None)  # d, lp, depth はすべて健全

    def test_h0_exceeded_mid_interval_alone(self):
        """区間内のどこかで H0.high を上抜けたが、評価日 T の Close/High は H0.high 未満
        （かつそのスパイクはまだ確定ラグ内で新しい確定スイング高値になっていない）。"""
        tail = np.array([220.0, 215.0, 210.0, 235.0, 220.0, 215.0])  # rel-index3 でスパイク
        close_full, h0 = build_leg(200.0, 230.0, tail, warm_start=195.0)
        high_s, low_s, close_s = _series(close_full)
        # peak 配列は h0 の後ろに3本続く（h0+1..h0+3）ので、tail[0] は h0+4 から始まる。
        # スパイクは tail の index3。
        spike_pos = h0 + 4 + 3
        T = spike_pos + 1  # スパイク翌日。スパイクの confirm_index = spike_pos+k > T
        r = compute(high_s, low_s, close_s, T)
        self.assertEqual(r["state"], STATE_INVALID)
        self.assertEqual(r["h0"], h0)  # スパイクはまだ新しい H0 として確定していない
        self.assertLessEqual(float(high_s.iloc[T]), r["h0_high"])  # 完了ではない
        self._assert_others_hold(r, None)


class ShallowDeepFlagsTest(unittest.TestCase):
    """浅押し(is_shallow: depth_pct<0.03)・深押し(is_deep: depth_pct>0.10)フラグ（§4.4末尾）。"""

    def test_is_shallow_when_depth_pct_below_3_percent(self):
        warmup = np.linspace(154.0, 180.0, 220)
        dip = np.array([180, 178, 176, 174, 176, 178, 180.0])
        rise = np.linspace(dip[-1], 220.0, 15)[1:]
        peak = np.array([227, 228, 229, 230, 229, 228, 227.0])  # H0直後の下落がごく浅い
        close_full = np.concatenate([warmup, dip, rise, peak])
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1  # h0(絶対位置244)の confirm_index(247) 当日
        r = compute(high_s, low_s, close_s, T)
        self.assertLess(r["depth_pct"], 0.03)
        self.assertTrue(r["is_shallow"])
        self.assertFalse(r["is_deep"])

    def test_neither_flag_when_depth_pct_between_thresholds(self):
        close_full, h0 = build_leg(200.0, 230.0, tail=np.array([220.0, 215.0]), warm_start=195.0)
        high_s, low_s, close_s = _series(close_full)
        T = len(close_full) - 1
        r = compute(high_s, low_s, close_s, T)
        self.assertTrue(0.03 <= r["depth_pct"] <= 0.10)
        self.assertFalse(r["is_shallow"])
        self.assertFalse(r["is_deep"])


class StructuralEdgeCasesTest(unittest.TestCase):
    def test_leg_bars_less_than_3_falls_back_to_no_structure(self):
        """外側足などで h0 と l0 が近すぎる（leg_bars<3）場合、例外を出さず構造なしに落ちる。"""
        raw = pd.DataFrame([
            (10, SWING_LOW, 90.0, 13),
            (12, SWING_HIGH, 110.0, 15),
        ], columns=["index", "kind", "value", "confirm_index"])
        alt = alternate_swings(raw)
        n = 30
        high_s, low_s, close_s = _series(np.linspace(100, 105, n))
        sma5_s = sma(close_s, 5)
        sma200_s = pd.Series(np.nan, index=close_s.index)
        atr14_s = atr_wilder(high_s, low_s, close_s, 14)
        r = pullback_state(high_s, low_s, close_s, sma5_s, sma200_s, atr14_s, alt, 20, k=2)
        self.assertEqual(r["state"], STATE_NO_STRUCTURE)
        self.assertIsNone(r["h0"])

    def test_h0_search_skips_lower_more_recent_confirmed_high(self):
        """直近の確定スイング高値が直前60本の最高値でなければ、それより古い有効な
        H0（真に60本の最高値であるもの）まで遡って選ぶこと（DESIGN.md §4.1）。"""
        warm_end = 180.0
        warmup = np.linspace(100.0, warm_end, 220)
        dip = np.array([180, 178, 176, 174, 176, 178, 180.0]) + (warmup[-1] - 180)
        rise = np.linspace(dip[-1], 220.0, 15)[1:]
        peak = np.array([220, 222, 224, 230, 224, 222, 220.0])  # 真のH0 (value 230, 絶対位置244)
        # 続けて下落し安値を作った後、60本条件を満たさない程度の二次的な山(value 221<230)を作る
        decline = np.array([217, 214, 211, 209])
        secondary_peak = np.array([212, 215, 218, 221, 218, 215, 212.0])
        tail = np.concatenate([decline, secondary_peak])
        close_full = np.concatenate([warmup, dip, rise, peak, tail])
        high_s, low_s, close_s = _series(close_full)
        h0_expected = len(warmup) + len(dip) + len(rise) + 3  # peak 配列の中央
        secondary_h0 = (len(warmup) + len(dip) + len(rise) + len(peak)
                        + len(decline) + 3)  # secondary_peak 配列の中央
        k = 3
        raw = detect_raw_swings(high_s, low_s, k)
        # secondary_h0 の方が新しい確定済みスイング高値だが、value(221.3) < 244 の value(230.3)
        highs = raw[raw["kind"] == SWING_HIGH]
        self.assertIn(secondary_h0, set(highs["index"]))
        self.assertLess(
            float(highs[highs["index"] == secondary_h0]["value"].iloc[0]),
            float(highs[highs["index"] == h0_expected]["value"].iloc[0]))

        alt = alternate_swings(raw)
        T = len(close_full) - 1
        self.assertGreaterEqual(T, secondary_h0 + k)  # secondary_h0 も確定済みであること
        h0 = find_h0(alt, high_s, T, k)
        self.assertEqual(h0, h0_expected,
                         f"{secondary_h0} は60本条件を満たさないため{h0_expected}まで遡るべき")


if __name__ == "__main__":
    unittest.main()
