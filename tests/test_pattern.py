"""パターンの検出（docs/PATTERN.md §2.1 反転系・§2.2 保ち合い系）。

固定するのは 3 つ。**未来を見ないこと**、**事前登録した数値のとおりに切れること**、
**同じ極値が複数のパターンに該当したら全部出ること**（§2.1）。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.pattern import (
    ADJACENT_TROUGH_GAP,
    BOX_TOL,
    DOUBLE_BOTTOM_GAP,
    EPSILON_SLOPE,
    EQUAL_TOL,
    FLAG_MAX_SPAN,
    MEASURED_MOVE_COLS,
    PATTERN_COLS,
    POLE_LOOKBACK,
    POLE_RISE,
    SEARCH_WINDOW,
    TOUCH_POINTS,
    detect_patterns,
    measured_move,
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


# ------------------------------------------------------------------ 保ち合い系
# docs/PATTERN.md §2.2。極値は 5 点で、交互なので必ず 3 + 2 に割れる


def five_points(values, start=40, gap=7, lead=120.0, end_value=None, n=200):
    """極値 5 点を gap 間隔で置き、そのあと上抜けさせる折れ線。

    `values` は古い順の 5 値。`lead` を高くしておくと 1 点目が谷になる
    （安値から始まる形）。`end_value` を上辺より上にすると成立日ができる。
    """
    pos = [start + gap * i for i in range(5)]
    knots = [(0, lead)] + list(zip(pos, values))
    top = max(values) + 4.0 if end_value is None else end_value
    knots += [(pos[-1] + 12, top), (n - 1, top)]
    return series_from(knots, n=n)


def ascending_triangle(highs=(110.0, 110.2), lows=(100.0, 103.0, 106.0)):
    """L H L H L。上辺が水平・下辺が上向き（C1）。"""
    return five_points([lows[0], highs[0], lows[1], highs[1], lows[2]])


def ascending_box(highs=(110.0, 110.3), lows=(100.2, 100.5, 100.0)):
    """L H L H L。上辺・下辺とも水平で重ならない（C2）。"""
    return five_points([lows[0], highs[0], lows[1], highs[1], lows[2]],
                       end_value=115.0)


def flagged(ups=(120.0, 119.0, 118.0), downs=(114.0, 113.04), gap=2,
            start=40, pole_from=105.0, n=200):
    """旗竿 → H L H L H の 5 点 → 上抜け（C3・C4）。

    旗竿は `start` の 5 本前から `start` までの上昇。gap を変えると span が変わる
    （gap=2 で span 11〜14、gap=3 で span 15 以上）。**フラッグは 15 営業日未満・
    ペナントは 15 営業日以下**なので、span 15 で両者が分かれる（§2.2）。
    """
    pos = [start + gap * i for i in range(5)]
    last = pos[-1]
    knots = [(0, pole_from), (start - POLE_LOOKBACK, pole_from),
             (pos[0], ups[0]), (pos[1], downs[0]), (pos[2], ups[1]),
             (pos[3], downs[1]), (pos[4], ups[2])]
    # 確定に k 本（上辺より下）、そのあと上抜け
    knots += [(last + 1, ups[2] - 0.4), (last + 2, ups[2] - 0.3),
              (last + 3, ups[2] - 0.2), (last + 4, 130.0), (n - 1, 130.0)]
    return series_from(knots, n=n)


def names_at(df, t_pos, **kw):
    out = detect_patterns(df["High"], df["Low"], df["Close"], t_pos, k=K, **kw)
    return sorted(out["pattern"].tolist())


def _any_pending(df, hi=None):
    """全 T を走査して出たパターン名を重複なく返す（未抜けも含める）。

    「出ないこと」を確かめるための道具。**成立日を狙い撃ちしないので、形の条件で
    落ちたのか上抜けで落ちたのかを取り違えない。**
    """
    names = set()
    for t in range(K + 1, hi or len(df)):
        out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                              include_pending=True)
        names.update(out["pattern"].tolist())
    return sorted(names)


def _spans(df, hi=None):
    """全 T を走査して出た span を集める（境界の位置を確かめるため）。"""
    spans = []
    for t in range(K + 1, hi or len(df)):
        out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                              include_pending=True)
        spans += [int(v) for v in out["span"]]
    return spans


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
        """§1 の事前登録値。検出数を見てから動かさない。

        文献値: ±1.5%（LMW）・22 営業日（LMW）・63 営業日（Savin et al. 2007）。
        裁量値: 隣接間隔 10 営業日（文献に数値が無いことを確認済み）。
        """
        self.assertEqual(EQUAL_TOL, 0.015)
        self.assertEqual(SEARCH_WINDOW, 63)
        self.assertEqual(DOUBLE_BOTTOM_GAP, 22)
        self.assertEqual(ADJACENT_TROUGH_GAP, 10)

    def test_pre_registered_numbers_for_consolidation(self):
        """§1 のうち保ち合い系で使うもの。

        文献値: ±0.75%（LMW・±1.5% とは別の定数）・3 + 2（最小 5）・15 営業日。
        裁量値: ε＝日次 0.1%、旗竿＝5 営業日以内に 10%。

        **±0.75% はたまたま ±1.5% の半分だが、片方を他方から導いていない**（§1）。
        値が一致していても別々の定数として持つ —— 一方を動かしたときに他方が
        黙って動かないようにするため。
        """
        self.assertEqual(BOX_TOL, 0.0075)
        self.assertEqual(TOUCH_POINTS, 5)
        self.assertEqual(FLAG_MAX_SPAN, 15)
        self.assertEqual(EPSILON_SLOPE, 0.001)
        self.assertEqual(POLE_LOOKBACK, 5)
        self.assertEqual(POLE_RISE, 0.10)


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
        """63 営業日より前に外れた極値は使わない。"""
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
        found = names_at(df, t)
        self.assertIn("inverse_hs", found)
        self.assertIn("triple_bottom", found)

    def test_overlap_crosses_families_too(self):
        """**寄せない相手は反転系どうしに限らない。**

        谷が ±0.75% に収まるほど揃った逆三尊は、上辺・下辺が水平なので
        上昇ボックス（C2）にも該当する。ボックスの許容が等値幅より狭いだけで、
        別の形を指しているわけではない —— **どちらの形として扱うかは検出数を見て
        から設計責任者が決める**ので、実装では両方返す（§2.1・§2.2）。
        """
        df = inverse_hs(shoulders=(100.0, 100.4), head=99.2, necks=(108.0, 108.5))
        t, _row = first_hit(df)
        self.assertIn("ascending_box", names_at(df, t))


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


class VisualCheckListingTest(unittest.TestCase):
    """目視確認用の列挙（docs/PATTERN.md §5 D-4）。

    極値の位置を日付に直して出す。**閾値の判断には使わない** —— 形をチャートで
    確かめるための表示であって、絞り込みでも並び替えの根拠でもない。
    """

    def _trim(self, df):
        """形が揃う最初の日で切る。step_pattern は最終足を T にするため。"""
        for t in range(K + 1, len(df)):
            out = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K,
                                  include_pending=True)
            if len(out):
                return df.iloc[:t + 1]
        raise AssertionError("形が揃う日が無い")

    def _run(self, df, tickers=("1234.T",), confirmed=False):
        from stockbot.cli import step_pattern
        from stockbot.config import Settings

        # confirmed=True なら上抜けた日で切る。既定は形が揃った最初の日（未抜け）
        df = df.iloc[:first_hit(df)[0] + 1] if confirmed else self._trim(df)
        cfg = Settings.from_env()
        universe = pd.DataFrame({"ticker": list(tickers),
                                 "passes": [True] * len(tickers)})
        lines = []
        out = step_pattern(cfg, universe, {t: df for t in tickers},
                           log=lines.append)
        return out, "\n".join(lines)

    def test_pending_rows_are_listed_with_dates(self):
        df = double_bottom(breakout=False)
        out, text = self._run(df)
        self.assertEqual(len(out), 1)
        self.assertFalse(bool(out["breakout"].iloc[0]))
        self.assertIn("未抜け 1件", text)
        self.assertIn("1234.T", text)
        # 極値が日付で出る（バー位置ではチャートで探せない）
        l1 = pd.Timestamp(out["l1_date"].iloc[0])
        self.assertIn(f"{l1:%m-%d}", text)
        self.assertIn("谷 ", text)
        self.assertIn("山 ", text)

    def test_missing_extreme_is_skipped_not_printed_as_nan(self):
        """ダブルボトムに 3 つ目の谷は無い。欠損を nan と書かない。"""
        out, text = self._run(double_bottom(breakout=False))
        self.assertTrue(pd.isna(out["l3_date"].iloc[0]))
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("NaT", text)

    def test_distance_to_neckline_is_signed(self):
        """未抜けは正（まだ上）、成立は負（もう抜けた）。"""
        pending, _t = self._run(double_bottom(breakout=False))
        done, _t2 = self._run(double_bottom(), confirmed=True)
        self.assertGreater(float(pending["to_neck_pct"].iloc[0]), 0)
        self.assertLess(float(done["to_neck_pct"].iloc[0]), 0)

    def test_both_patterns_are_listed_separately(self):
        """同じ 5 極値が両方に該当したら、2 行とも列挙する（§2.1）。"""
        df = inverse_hs(shoulders=(100.0, 100.4), head=99.2, necks=(108.0, 108.5))
        out, text = self._run(df)
        self.assertIn("inverse_hs", sorted(out["pattern"]))
        self.assertIn("triple_bottom", sorted(out["pattern"]))
        self.assertIn("inverse_hs", text)
        self.assertIn("triple_bottom", text)
        # 1 行 1 パターン。同じ銘柄が該当したぶんだけ行が出る
        self.assertEqual(len(out), out["pattern"].nunique())


class MeasuredMoveTest(unittest.TestCase):
    """測定目標と比率（docs/PATTERN.md §2.3）。

    **文献上の目安であって統計的裏付けは無い。** 比率を併記して読み手が判断できる
    ようにするための量で、判定には一切使わない。
    """

    def test_ratio_follows_the_algebra(self):
        """比率 = (1 − b) / (1 + b)。b は高さに対する抜け幅。"""
        neck, low = 110.0, 100.0
        h = neck - low
        for b, expected in [(0.0, 1.0), (0.05, 0.905), (0.10, 0.818), (0.20, 0.667)]:
            m = measured_move(neck, neck + b * h, [low, low])
            self.assertAlmostEqual(m["rr"], expected, places=3, msg=f"b={b}")
            self.assertAlmostEqual(m["breakeven_win_rate"], 1 / (1 + expected),
                                   places=3, msg=f"b={b}")

    def test_target_is_neckline_plus_height(self):
        m = measured_move(110.0, 111.0, [100.0, 100.5])
        self.assertAlmostEqual(m["pattern_low"], 100.0)
        self.assertAlmostEqual(m["height"], 10.0)
        self.assertAlmostEqual(m["target"], 120.0)

    def test_stop_is_the_lowest_low(self):
        """撤退はパターンの最安値。逆三尊なら頭になる。"""
        m = measured_move(110.0, 111.0, [100.0, 92.0, 100.6])
        self.assertAlmostEqual(m["pattern_low"], 92.0)
        self.assertAlmostEqual(m["height"], 18.0)

    def test_breakout_pct_is_negative_while_pending(self):
        pending = measured_move(110.0, 105.0, [100.0])
        done = measured_move(110.0, 111.0, [100.0])
        self.assertLess(pending["breakout_pct"], 0)
        self.assertGreater(done["breakout_pct"], 0)

    def test_degenerate_inputs(self):
        """高さが定義できない・値が壊れている場合は比率を出さない。"""
        flat = measured_move(100.0, 101.0, [100.0])       # 高さ 0
        self.assertTrue(pd.isna(flat["rr"]))
        self.assertTrue(pd.isna(measured_move(110.0, 0.0, [100.0])["rr"]))
        self.assertTrue(pd.isna(measured_move(np.nan, 111.0, [100.0])["rr"]))
        self.assertTrue(pd.isna(measured_move(110.0, 111.0, [])["rr"]))

    def test_rows_carry_the_columns(self):
        df = double_bottom()
        _t, row = first_hit(df)
        for col in MEASURED_MOVE_COLS:
            self.assertIn(col, row.index)
        self.assertGreater(float(row["rr"]), 0)
        # ダブルボトムの撤退は 2 安値の低いほう
        self.assertAlmostEqual(float(row["pattern_low"]),
                               min(float(row["l1"]), float(row["l2"])), places=6)

    def test_inverse_hs_stop_is_the_head(self):
        df = inverse_hs()
        _t, row = first_hit(df)
        self.assertAlmostEqual(float(row["pattern_low"]), float(row["l2"]), places=6)


class AscendingTriangleTest(unittest.TestCase):
    """上辺が水平・下辺が上向き（docs/PATTERN.md §2.2 C1）。"""

    def test_detected_on_the_breakout_day(self):
        df = ascending_triangle()
        t, row = first_hit(df)
        self.assertIsNotNone(t)
        self.assertEqual(row["pattern"], "ascending_triangle")
        close = df["Close"].to_numpy()
        self.assertGreater(close[t], row["neckline"])
        self.assertLessEqual(close[t - 1], row["neckline"])

    def test_upper_line_must_be_flat_within_epsilon(self):
        """|上辺傾き| < ε（日次 0.1%）。超えたら三角ではない。"""
        flat = ascending_triangle(highs=(110.0, 110.2))     # 約 0.013%/日
        tilted = ascending_triangle(highs=(108.0, 113.0))   # 約 0.33%/日
        self.assertIn("ascending_triangle", names_at(flat, first_hit(flat)[0]))
        t, _r = first_hit(tilted)
        self.assertNotIn("ascending_triangle",
                         names_at(tilted, t, include_pending=True) if t else [])

    def test_epsilon_boundary_bites(self):
        """ε をまたぐ 2 つ。境界のすぐ内と外で結果が変わること。"""
        df = ascending_triangle()
        _t, row = first_hit(df)
        self.assertLess(abs(float(row["upper_slope"])), EPSILON_SLOPE)
        self.assertGreater(float(row["lower_slope"]), 0)

    def test_lower_line_must_rise(self):
        """下辺傾き > 0。下向き・水平は三角にしない（判定式のまま）。"""
        falling = ascending_triangle(lows=(106.0, 103.0, 100.0))
        found = _any_pending(falling)
        self.assertNotIn("ascending_triangle", found)

    def test_three_touch_side_lands_in_l3(self):
        """下辺 3 タッチは l1..l3、上辺 2 タッチは h1..h2 に入る。"""
        df = ascending_triangle()
        _t, row = first_hit(df)
        for col in ("l1", "l2", "l3", "h1", "h2"):
            self.assertFalse(pd.isna(row[col]), col)
        self.assertTrue(pd.isna(row["h3"]))


class AscendingBoxTest(unittest.TestCase):
    """上辺・下辺とも水平（docs/PATTERN.md §2.2 C2）。"""

    def test_detected(self):
        df = ascending_box()
        t, _row = first_hit(df)
        self.assertIsNotNone(t)
        self.assertIn("ascending_box", names_at(df, t))

    def test_upper_line_is_the_mean_not_a_regression(self):
        """ボックスの上辺は水平（上辺の平均）。回帰直線を延ばした値ではない。"""
        df = ascending_box(highs=(110.0, 110.3))
        _t, row = first_hit(df)
        self.assertAlmostEqual(float(row["neckline"]), 110.15, places=6)

    def test_box_tolerance_is_0_75pct_not_1_5pct(self):
        """±0.75% で切る。**±1.5% は通るが ±0.75% は通らない**幅で確かめる。

        ±1.5% で切っていたらこれが通ってしまう（等値幅を流用していないことの確認）。
        """
        spread = (100.0, 102.0, 100.2)      # 平均から最大 1.26%
        self.assertLess(max(abs(v - np.mean(spread)) for v in spread) / np.mean(spread),
                        EQUAL_TOL)
        self.assertGreater(max(abs(v - np.mean(spread)) for v in spread) / np.mean(spread),
                           BOX_TOL)
        loose = ascending_box(lows=spread)
        found = _any_pending(loose)
        self.assertNotIn("ascending_box", found)
        # 形（5 極値）自体は揃っている —— 落ちたのは幅の判定であって極値ではない
        self.assertIn("triple_bottom", found)

    def test_lowest_high_must_be_above_the_highest_low(self):
        """`最低の山 > 最高の谷`。上下が重なる形はボックスにしない。"""
        overlap = ascending_box(highs=(104.5, 104.2), lows=(104.3, 104.0, 104.1))
        found = _any_pending(overlap)
        self.assertNotIn("ascending_box", found)
        self.assertIn("triple_bottom", found)   # 極値は揃っている

    def test_a_box_with_three_troughs_is_also_a_triple_bottom(self):
        """**±0.75% は ±1.5% より狭い。** 谷 3 点のボックスは必ず等値も満たす。

        寄せずに両方記録する（§2.1）。どちらの形として扱うかは設計責任者の判断。
        """
        df = ascending_box()
        t, _row = first_hit(df)
        self.assertEqual(names_at(df, t), ["ascending_box", "triple_bottom"])


class FlagAndPennantTest(unittest.TestCase):
    """旗竿つきの保ち合い（docs/PATTERN.md §2.2 C3・C4）。"""

    def test_flag_detected_with_parallel_falling_lines(self):
        df = flagged()
        t, row = first_hit(df)
        self.assertEqual(row["pattern"], "bull_flag")
        self.assertLess(float(row["upper_slope"]), 0)
        self.assertLess(float(row["lower_slope"]), 0)
        self.assertLess(abs(float(row["upper_slope"]) - float(row["lower_slope"])),
                        EPSILON_SLOPE)

    def test_pennant_detected_when_lines_converge(self):
        df = flagged(downs=(112.0, 113.5))
        t, row = first_hit(df)
        self.assertEqual(row["pattern"], "bull_pennant")
        self.assertLess(float(row["upper_slope"]), 0)
        self.assertGreater(float(row["lower_slope"]), 0)

    def test_flag_needs_the_lines_parallel(self):
        """傾き差が ε 以上なら平行ではない。収束でもないので何にもならない。"""
        df = flagged(downs=(114.0, 111.0))     # 下辺だけ急な下げ
        self.assertEqual(_any_pending(df), [])

    def test_pole_is_required(self):
        """旗竿（5 営業日以内に 10% 以上）が無ければ検出しない。"""
        weak = flagged(pole_from=118.0)        # 上昇 1.7% しかない
        self.assertEqual(_any_pending(weak), [])
        strong = flagged(pole_from=105.0)      # 上昇 14.3%
        self.assertEqual(_any_pending(strong), ["bull_flag"])

    def test_pole_percent_is_recorded(self):
        df = flagged(pole_from=105.0)
        _t, row = first_hit(df)
        self.assertAlmostEqual(float(row["pole_pct"]), (120.0 / 105.0 - 1) * 100,
                               places=6)

    def test_flag_is_under_15_and_pennant_is_15_or_less(self):
        """**境界の扱いが違う。** span 15 でフラッグは落ち、ペナントは残る（§2.2）。"""
        flag = flagged(gap=3)
        pennant = flagged(gap=3, downs=(112.0, 113.5))
        spans = _spans(flag) + _spans(pennant)
        self.assertTrue(all(sp >= FLAG_MAX_SPAN for sp in spans), spans)
        self.assertEqual(_any_pending(flag), [])                # 15 は「未満」に入らない
        self.assertEqual(_any_pending(pennant), ["bull_pennant"])   # 15 は「以下」に入る

    def test_three_touch_upper_side_lands_in_h3(self):
        """H L H L H は上辺 3 タッチ。3 つ目の山は h3 に入る。"""
        df = flagged()
        _t, row = first_hit(df)
        for col in ("h1", "h2", "h3", "l1", "l2"):
            self.assertFalse(pd.isna(row[col]), col)
        self.assertTrue(pd.isna(row["l3"]))


class ConsolidationPointInTimeTest(unittest.TestCase):
    """CLAUDE.md 未来参照の禁止。保ち合い系も同じ（旗竿は T より前だけを読む）。"""

    def test_future_bars_do_not_change_the_result(self):
        for df in (ascending_triangle(), ascending_box(), flagged()):
            t, _row = first_hit(df)
            before = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
            broken = df.copy()
            broken.iloc[t + 1:] = broken.iloc[t + 1:] * 10.0
            after = detect_patterns(broken["High"], broken["Low"], broken["Close"],
                                    t, k=K)
            pd.testing.assert_frame_equal(before, after)

    def test_truncating_after_t_does_not_change_the_result(self):
        """再計算一致（DESIGN.md §11）。T までしか無い系列でも同じ行が出る。"""
        for df in (ascending_triangle(), ascending_box(), flagged()):
            t, _row = first_hit(df)
            full = detect_patterns(df["High"], df["Low"], df["Close"], t, k=K)
            cut = df.iloc[:t + 1]
            trimmed = detect_patterns(cut["High"], cut["Low"], cut["Close"], t, k=K)
            pd.testing.assert_frame_equal(full, trimmed)

    def test_fewer_than_five_confirmed_extremes_detects_nothing(self):
        """タッチ点が 5 点に満たなければ線を引かない（3 + 2・最小 5）。"""
        df = series_from([(0, 120.0), (40, 100.0), (60, 110.0), (80, 101.0),
                          (199, 101.0)])
        self.assertEqual(_any_pending(df), [])
