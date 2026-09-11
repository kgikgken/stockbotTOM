"""画像カードの表示内容（docs/PATTERN.md §6）。

固定するのは 4 つ。**パターン別にセクションが分かれること**、**成立0件の日も出る
こと**、**成立した銘柄は監視から外れること**、**値が記録の列から来ること**。

Playwright を使わないので、表示内容のテストはここで完結する（§4.3）。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.render.context import (
    NO_DONE_NOTE,
    PATTERN_LABELS,
    PATTERN_ORDER,
    TARGET_NOTE,
    WATCH_MAX,
    build_card,
    build_context,
)
from stockbot.render.render import render_html


def row(ticker="5202.T", pattern="inverse_hs", **kw):
    base = {
        "ticker": ticker, "name": "日本板硝子", "sector33": "ガラス・土石製品",
        "pattern": pattern, "close_t": 495.0, "neckline": 493.0, "breakout_pct": 0.41,
        "pattern_low": 478.0, "height": 15.0, "target": 508.0,
        "up_pct": 2.6, "down_pct": -3.4, "rr": 0.76, "span": 37,
        "l1_date": pd.Timestamp("2026-07-17"), "l1": 480.0,
        "l2_date": pd.Timestamp("2026-08-12"), "l2": 478.0,
        "l3_date": pd.Timestamp("2026-09-04"), "l3": 490.0,
        "h1_date": pd.Timestamp("2026-07-30"), "h1": 484.0,
        "h2_date": pd.Timestamp("2026-09-01"), "h2": 493.0,
        "h3_date": pd.NaT, "h3": np.nan,
        "upper_slope": np.nan, "lower_slope": np.nan, "pole_pct": np.nan,
        "upper_scatter": np.nan,
        "adv_jpy": 5.2e8, "earnings_days": np.nan, "earnings_unknown": True,
    }
    base.update(kw)
    return base


def frame(rows):
    return pd.DataFrame(rows) if rows else None


def summary(**kw):
    out = {"delivered_on": "2026-09-14", "asof": "2026-09-11",
           "n_evaluated": 1277, "n_watch": 196,
           "counts": {"inverse_hs": {"done": 1, "watch": 2},
                      "ascending_triangle": {"done": 11, "watch": 172}}}
    out.update(kw)
    return out


class SectionTest(unittest.TestCase):
    """パターン別にセクションを分け、中を候補（成立）と監視（未抜け）に分ける（§6.1）。"""

    def test_split_by_pattern_in_document_order(self):
        done = frame([row("1.T", "ascending_triangle"), row("2.T", "inverse_hs")])
        ctx = build_context(done, None, summary())
        labels = [s["pattern"] for s in ctx["sections"]]
        self.assertEqual(labels, ["inverse_hs", "ascending_triangle"])   # §2 の並び
        self.assertLess(PATTERN_ORDER.index("inverse_hs"),
                        PATTERN_ORDER.index("ascending_triangle"))

    def test_done_and_watch_are_separate_inside_a_section(self):
        done = frame([row("1.T", "inverse_hs")])
        watch = frame([row("2.T", "inverse_hs")])
        ctx = build_context(done, watch, summary())
        sec = ctx["sections"][0]
        self.assertEqual([c["ticker"] for c in sec["done"]], ["1.T"])
        self.assertEqual([c["ticker"] for c in sec["watch"]], ["2.T"])
        self.assertFalse(sec["done"][0]["watch"])
        self.assertTrue(sec["watch"][0]["watch"])

    def test_unknown_pattern_name_is_not_dropped(self):
        ctx = build_context(frame([row("1.T", "brand_new")]), None, summary())
        self.assertEqual([s["pattern"] for s in ctx["sections"]], ["brand_new"])

    def test_all_seven_patterns_have_a_japanese_label(self):
        self.assertEqual(len(PATTERN_LABELS), 7)
        for name in PATTERN_ORDER:
            self.assertTrue(PATTERN_LABELS[name])


class ZeroDoneTest(unittest.TestCase):
    """**成立0件の日も配信する**（§6.1）。監視だけを出し、0件であることを明記する。"""

    def test_note_appears_only_when_there_is_no_done(self):
        ctx = build_context(None, frame([row()]), summary())
        self.assertEqual(ctx["no_done_note"], NO_DONE_NOTE)
        ctx2 = build_context(frame([row()]), None, summary())
        self.assertIsNone(ctx2["no_done_note"])

    def test_watch_is_still_shown(self):
        ctx = build_context(None, frame([row("2331.T", "ascending_triangle")]), summary())
        self.assertEqual(ctx["n_done"], 0)
        self.assertEqual(ctx["sections"][0]["watch"][0]["ticker"], "2331.T")

    def test_nothing_at_all(self):
        ctx = build_context(None, None, summary(n_watch=0, counts={}))
        self.assertEqual(ctx["sections"], [])
        self.assertEqual(ctx["no_done_note"], NO_DONE_NOTE)


class WatchTest(unittest.TestCase):
    def test_done_ticker_is_dropped_from_watch(self):
        """5202.T は逆三尊が成立・上昇三角が未抜けだった（2026-09-10 の実データ）。

        **成立を優先して監視から外す**（§6.4）。記録は成立のみなので表示だけの話。
        """
        done = frame([row("5202.T", "inverse_hs")])
        watch = frame([row("5202.T", "ascending_triangle"),
                       row("2331.T", "ascending_triangle")])
        ctx = build_context(done, watch, summary())
        shown = [(c["ticker"], c["watch"]) for s in ctx["sections"] for c in s["watch"]]
        self.assertEqual(shown, [("2331.T", True)])

    def test_capped_at_watch_max_across_all_patterns(self):
        """**全体で上位 10 件**。パターンごとに 10 件ずつにすると 1 枚に収まらない。"""
        watch = frame([row(f"{1000 + i}.T", "ascending_triangle")
                       for i in range(WATCH_MAX + 5)])
        ctx = build_context(None, watch, summary())
        self.assertEqual(ctx["n_watch_shown"], WATCH_MAX)

    def test_given_order_is_preserved(self):
        """並びは呼び出し側（上値抵抗線に近い順）のまま。ここで並べ替えない。"""
        watch = frame([row("3.T", "ascending_triangle"), row("1.T", "ascending_triangle"),
                       row("2.T", "ascending_triangle")])
        ctx = build_context(None, watch, summary())
        self.assertEqual([c["ticker"] for c in ctx["sections"][0]["watch"]],
                         ["3.T", "1.T", "2.T"])

    def test_streak_only_on_watch_and_only_after_the_first_day(self):
        watch = frame([row("1.T", "ascending_triangle", watch_streak=4)])
        ctx = build_context(None, watch, summary())
        self.assertEqual(ctx["sections"][0]["watch"][0]["streak"], "監視4日目")
        first = build_context(None, frame([row(watch_streak=1)]), summary())
        self.assertIsNone(first["sections"][0]["watch"][0]["streak"])
        # 成立側には出さない
        done = build_context(frame([row(watch_streak=4)]), None, summary())
        self.assertIsNone(done["sections"][0]["done"][0]["streak"])


class CardTest(unittest.TestCase):
    """カードの項目（§6.2）。値はすべて記録の列から来る。"""

    def test_required_fields(self):
        c = build_card(pd.Series(row()))
        self.assertEqual(c["ticker"], "5202.T")
        self.assertEqual(c["pattern_label"], "逆三尊")
        self.assertEqual(c["sector33"], "ガラス・土石製品")
        self.assertEqual(c["close"], "495.0")
        self.assertEqual(c["neckline"], "493.0")
        self.assertEqual(c["breakout"], "+0.41%")
        self.assertEqual(c["stop"], "478.0")
        self.assertEqual(c["target"], "508.0")
        self.assertEqual(c["rr"], "0.76")
        self.assertEqual(c["span"], "37本")
        self.assertEqual(c["adv"], "5.2億円")

    def test_extremes_carry_dates(self):
        """**極値の日付を載せる。** 押し目型と決定的に違う点（§6.2）。"""
        c = build_card(pd.Series(row()))
        self.assertEqual([e["date"] for e in c["troughs"]], ["07-17", "08-12", "09-04"])
        self.assertEqual([e["date"] for e in c["peaks"]], ["07-30", "09-01"])
        self.assertEqual(c["peaks"][0]["value"], "484.0")

    def test_extremes_accept_iso_strings_from_the_snapshot(self):
        """監視はスナップショット（JSON）経由なので日付が文字列で来る。"""
        c = build_card(pd.Series(row(l1_date="2026-07-17", h1_date="2026-07-30")))
        self.assertEqual(c["troughs"][0]["date"], "07-17")

    def test_missing_extreme_is_skipped_not_printed(self):
        c = build_card(pd.Series(row(l3_date=pd.NaT, l3=np.nan)))
        self.assertEqual(len(c["troughs"]), 2)

    def test_neckline_label_differs_by_family(self):
        self.assertEqual(build_card(pd.Series(row(pattern="inverse_hs")))["neckline_label"],
                         "ネックライン")
        self.assertEqual(
            build_card(pd.Series(row(pattern="ascending_triangle")))["neckline_label"],
            "上値抵抗線")

    def test_scatter_only_when_present_and_flagged_over_the_box_tolerance(self):
        """山 3 点以上のときだけ出す。±0.75% 超は色を変える（§2.2・D-13）。"""
        self.assertIsNone(build_card(pd.Series(row()))["scatter"])
        wide = build_card(pd.Series(row(upper_scatter=2.93)))
        self.assertEqual(wide["scatter"], "2.93%")
        self.assertTrue(wide["scatter_wide"])
        narrow = build_card(pd.Series(row(upper_scatter=0.48)))
        self.assertFalse(narrow["scatter_wide"])

    def test_earnings_unknown_is_marked(self):
        self.assertTrue(build_card(pd.Series(row()))["earnings"]["unknown"])
        known = build_card(pd.Series(row(earnings_days=5.0, earnings_unknown=False)))
        self.assertEqual(known["earnings"]["text"], "決算まで5営業日")
        self.assertFalse(known["earnings"]["unknown"])

    def test_missing_values_are_dashes_not_nan(self):
        c = build_card(pd.Series(row(target=np.nan, rr=np.nan, adv_jpy=np.nan)))
        self.assertEqual(c["target"], "—")
        self.assertEqual(c["rr"], "—")
        self.assertEqual(c["adv"], "—")


class BreakdownTest(unittest.TestCase):
    """2枚目の内訳（§6.3）。表の合計は表の行から出す（食い違わせない）。"""

    def test_totals_match_the_rows(self):
        ctx = build_context(frame([row()]), None, summary())
        self.assertEqual(ctx["breakdown_done"], sum(b["done"] for b in ctx["breakdown"]))
        self.assertEqual(ctx["breakdown_watch"], sum(b["watch"] for b in ctx["breakdown"]))

    def test_zero_rows_are_omitted(self):
        ctx = build_context(None, None, summary(counts={"bull_flag": {"done": 0, "watch": 0}}))
        self.assertEqual(ctx["breakdown"], [])


class HtmlTest(unittest.TestCase):
    """テンプレートまで通す（Playwright は使わない）。"""

    def _html(self, done, watch, **kw):
        return render_html(done, watch, summary(**kw))

    def test_renders_both_pages(self):
        html = self._html(frame([row()]), frame([row("2331.T", "ascending_triangle")]))
        self.assertIn('id="page1"', html)
        self.assertIn('id="page2"', html)

    def test_shows_the_pattern_and_the_extremes(self):
        html = self._html(frame([row()]), None)
        for token in ("逆三尊", "5202.T", "07-17", "測定目標（文献上の目安）", "撤退の目安"):
            self.assertIn(token, html)

    def test_zero_done_day_says_so(self):
        html = self._html(None, frame([row()]))
        self.assertIn(NO_DONE_NOTE, html)

    def test_notes_are_present(self):
        html = self._html(frame([row()]), None)
        self.assertIn(TARGET_NOTE, html)
        self.assertIn("業種は表示のみ", html)

    def test_no_markdown_emphasis_leaks_into_the_page(self):
        """注記はそのまま HTML に入る。`**` を書くと文字として出る。"""
        html = self._html(frame([row()]), frame([row("2.T", "ascending_triangle")]))
        body = html.split('id="page1"')[1]
        self.assertNotIn("**", body)

    def test_no_nan_in_the_output(self):
        html = self._html(frame([row(target=np.nan, rr=np.nan)]), None)
        self.assertNotIn("nan", html.split("<style>")[1].lower().split("</style>")[1])
        self.assertNotIn("NaT", html)


if __name__ == "__main__":
    unittest.main()
