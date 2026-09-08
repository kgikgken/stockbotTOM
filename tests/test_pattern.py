"""反転系パターンの検出（docs/PATTERN.md §2.1）。

固定するのは 3 つ。**未来を見ないこと**、**事前登録した数値のとおりに切れること**、
**同じ 5 極値が R2 と R3 の両方に該当したら両方出ること**（§2.1）。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.pattern import (
    ADJACENT_TROUGH_GAP,
    DOUBLE_BOTTOM_GAP,
    EQUAL_TOL,
    PATTERN_COLS,
    SEARCH_WINDOW,
    detect_patterns,
)

K = 3   # swings.py の確定ラグ。PATTERN.md §2.1 共通


def series_from(points, n=200, base=100.0):
    """(位置, 値) の折れ線から日足を作る。High/Low は Close と同じにする。

    スイング検出は High/Low を見るので、極値をそのまま置けば意図した形になる。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    idx = np.arange(n)
    close = np.interp(idx, xs, ys)
    dates = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close},
                        index=dates)


def double_bottom(gap=30, neck=110.0, l1=100.0, l2=100.5, breakout=True):
    """L1 → H → L2 → 上抜け。gap は L1 と L2 の距離。"""
    a, b = 40, 40 + gap
    end = b + 12
    top = neck + 3.0 if breakout else neck - 3.0
    return series_from([(0, 120.0), (a, l1), ((a + b) // 2, neck), (b, l2),
                        (end, top), (199, top)])


def triple_bottom(gap=14, lows=(100.0, 100.4, 99.8), necks=(108.0, 110.0)):
    a = 40
    b, c = a + gap, a + 2 * gap
    end = c + 12
    return series_from([(0, 125.0), (a, lows[0]), ((a + b) // 2, necks[0]),
                        (b, lows[1]), ((b + c) // 2, necks[1]), (c, lows[2]),
                        (end, max(necks) + 3.0), (199, max(necks) + 3.0)])


def inverse_hs(gap=14, shoulders=(100.0, 100.6), head=92.0, necks=(108.0, 108.5)):
    a = 40
    b, c = a + gap, a + 2 * gap
    end = c + 12
    return series_from([(0, 125.0), (a, shoulders[0]), ((a + b) // 2, necks[0]),
                        (b, head), ((b + c) // 2, necks[1]), (c, shoulders[1]),
                        (end, max(necks) + 3.0), (199, max(necks) + 3.0)])


def names_at(df, t_pos, **kw):
    out = detect_patterns(df["High"], df["Low"], df["Close"], t_pos, k=K, **kw)
    return sorted(out["pattern"].tolist())


def first_hit(df, **kw):
    """最初に検出された (t_pos, 行) を返す。出なければ (None, None)。"""
    for t in range(K + 1, len(df)):
        out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K, **kw)
        if len(out):
            return t, out.iloc[0]
    return None, None


class ContractTest(unittest.TestCase):
    def test_columns_and_empty(self):
        df = series_from([(0, 100.0), (199, 100.0)])
        out = detect_patterns(df["High"], df["Low"], df["Close"], 150, k=K)
        self.assertEqual(list(out.columns), PATTERN_COLS)
        self.assertEqual(len(out), 0)

    def test_pre_registered_numbers(self):
        """§1 の事前登録値。検出数を見てから動かさない。"""
        self.assertEqual(EQUAL_TOL, 0.015)
        self.assertEqual(SEARCH_WINDOW, 60)
        self.assertEqual(DOUBLE_BOTTOM_GAP, 22)
        self.assertEqual(ADJACENT_TROUGH_GAP, 10)


class DoubleBottomTest(unittest.TestCase):
    def test_detected_on_the_breakout_day(self):
        df = double_bottom()
        t, row = first_hit(df)
        self.assertIsNotNone(t)
        self.assertEqual(row["pattern"], "double_bottom")
        # 成立日は終値がネックラインを上抜けた最初の日
        close = df["Close"].to_numpy()
        self.assertGreater(close[t], row["neckline"])
        self.assertLessEqual(close[t - 1], row["neckline"])

    def test_not_detected_before_the_breakout(self):
        """極値が揃っただけでは成立しない（§2.1 共通）。"""
        df = double_bottom(breakout=False)
        self.assertIsNone(first_hit(df)[0])

    def test_equal_tolerance_boundary(self):
        """±1.5% を超える 2 安値は等値と認めない。"""
        inside = double_bottom(l1=100.0, l2=101.4)     # 1.39%
        outside = double_bottom(l1=100.0, l2=103.1)    # 3.05%
        self.assertIsNotNone(first_hit(inside)[0])
        self.assertIsNone(first_hit(outside)[0])

    def test_minimum_gap_is_22_business_days(self):
        self.assertIsNotNone(first_hit(double_bottom(gap=24))[0])
        self.assertIsNone(first_hit(double_bottom(gap=14))[0])

    def test_search_window_excludes_old_extremes(self):
        """60 営業日より前に外れた極値は使わない。"""
        df = double_bottom(gap=30)
        t, _row = first_hit(df)
        # 同じ形でも、成立日をずっと後ろにすると極値がウィンドウから外れる
        late = detect_patterns(df["High"], df["Low"], df["Close"],
                               t + SEARCH_WINDOW + 5, k=K)
        self.assertEqual(len(late), 0)


class TripleBottomTest(unittest.TestCase):
    def test_detected(self):
        _t, row = first_hit(triple_bottom())
        self.assertIsNotNone(row is not None or None)
        self.assertIn("triple_bottom", names_at(triple_bottom(), _t))
        # ネックラインは谷間の高値 2 点のうち高いほう
        self.assertAlmostEqual(float(row["neckline"]), 110.0, places=6)

    def test_each_trough_within_1_5pct_of_the_mean(self):
        ok = triple_bottom(lows=(100.0, 101.0, 99.5))
        ng = triple_bottom(lows=(100.0, 106.0, 99.5))
        self.assertIn("triple_bottom", names_at(ok, first_hit(ok)[0]))
        t, _r = first_hit(ng)
        self.assertNotIn("triple_bottom", names_at(ng, t) if t else [])

    def test_adjacent_gap_is_10_business_days(self):
        wide = triple_bottom(gap=12)
        narrow = triple_bottom(gap=6)
        self.assertIn("triple_bottom", names_at(wide, first_hit(wide)[0]))
        self.assertIsNone(first_hit(narrow)[0])


class InverseHeadShouldersTest(unittest.TestCase):
    def test_detected(self):
        df = inverse_hs()
        t, row = first_hit(df)
        self.assertIsNotNone(t)
        self.assertIn("inverse_hs", names_at(df, t))
        self.assertAlmostEqual(float(row["neckline"]), 108.5, places=6)

    def test_head_must_be_below_both_shoulders(self):
        """頭が両肩より低くなければ逆三尊ではない。深さの下限は無い。"""
        shallow = inverse_hs(head=99.0)        # わずかに低いだけでも成立
        raised = inverse_hs(head=100.3)        # 肩の間に入ってしまう
        self.assertIn("inverse_hs", names_at(shallow, first_hit(shallow)[0]))
        t, _r = first_hit(raised)
        self.assertNotIn("inverse_hs", names_at(raised, t) if t else [])

    def test_shoulders_within_1_5pct(self):
        ok = inverse_hs(shoulders=(100.0, 101.0))
        ng = inverse_hs(shoulders=(100.0, 106.0))
        self.assertIn("inverse_hs", names_at(ok, first_hit(ok)[0]))
        t, _r = first_hit(ng)
        self.assertNotIn("inverse_hs", names_at(ng, t) if t else [])

    def test_neckline_points_within_1_5pct(self):
        ok = inverse_hs(necks=(108.0, 108.5))
        ng = inverse_hs(necks=(104.0, 118.0))
        self.assertIn("inverse_hs", names_at(ok, first_hit(ok)[0]))
        t, _r = first_hit(ng)
        self.assertNotIn("inverse_hs", names_at(ng, t) if t else [])


class BothPatternsTest(unittest.TestCase):
    def test_same_extremes_can_be_both_r2_and_r3(self):
        """3 谷が等値かつ中央が両側より低い形は、両方を記録する（§2.1）。"""
        df = inverse_hs(shoulders=(100.0, 100.4), head=99.2, necks=(108.0, 108.5))
        t, _row = first_hit(df)
        self.assertIsNotNone(t)
        self.assertEqual(names_at(df, t), ["inverse_hs", "triple_bottom"])


class PendingTest(unittest.TestCase):
    """形は揃ったがネックライン未抜けの行（docs/PATTERN.md §2.1）。

    検出 0 件だったときの切り分け用。**成立の定義は変えていない** ——
    既定では成立した行だけが返る。
    """

    def test_pending_row_appears_only_when_asked(self):
        df = double_bottom(breakout=False)
        # 形が揃う日を探す（上抜けは起きない系列）
        found = None
        for t in range(K + 1, len(df)):
            out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                                  include_pending=True)
            if len(out):
                found = (t, out)
                break
        self.assertIsNotNone(found, "形が揃う日が無い")
        t, out = found
        self.assertFalse(bool(out["breakout"].iloc[0]))
        # 既定では返らない
        self.assertEqual(len(detect_patterns(df["High"], df["Low"], df["Close"],
                                             t, k=K)), 0)

    def test_confirmed_rows_have_breakout_true(self):
        df = double_bottom()
        t, _row = first_hit(df)
        out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
        self.assertTrue(bool(out["breakout"].iloc[0]))

    def test_pending_includes_confirmed(self):
        """include_pending は成立行も落とさない（合計＝成立＋未抜け）。"""
        df = double_bottom()
        t, _row = first_hit(df)
        both = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                               include_pending=True)
        only = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
        self.assertEqual(len(both), len(only))
        self.assertTrue(bool(both["breakout"].iloc[0]))

    def test_shape_conditions_still_apply_to_pending(self):
        """未抜けでも形の条件は同じ。間隔が足りなければ行は出ない。"""
        df = double_bottom(gap=14, breakout=False)
        for t in range(K + 1, len(df)):
            out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                                  include_pending=True)
            self.assertEqual(len(out), 0)


class PointInTimeTest(unittest.TestCase):
    """CLAUDE.md 未来参照の禁止。"""

    def test_future_bars_do_not_change_the_result(self):
        df = double_bottom()
        t, _row = first_hit(df)
        before = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
        broken = df.copy()
        broken.iloc[t + 1:] = broken.iloc[t + 1:] * 10.0   # T より後だけ壊す
        after = detect_patterns(broken["High"], broken["Low"], broken["Close"], t, k=K)
        pd.testing.assert_frame_equal(before, after)

    def test_truncating_after_t_does_not_change_the_result(self):
        """T までしか無い系列でも同じ結果になる（再計算一致）。"""
        df = double_bottom()
        t, _row = first_hit(df)
        full = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
        cut = df.iloc[:t + 1]
        trimmed = detect_patterns(cut["High"], cut["Low"], cut["Close"], t, k=K)
        pd.testing.assert_frame_equal(full, trimmed)

    def test_unconfirmed_swings_are_not_used(self):
        """確定ラグ k 本を満たさない極値は使わない。"""
        df = double_bottom()
        t, row = first_hit(df)
        # 直近の安値は confirm_index <= t を満たしているはず
        self.assertLessEqual(int(row["l2_pos"]) + K, t)


if __name__ == "__main__":
    unittest.main()
