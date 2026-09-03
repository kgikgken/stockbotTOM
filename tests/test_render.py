"""画像カードの表示内容（docs/SCREENER.md §4.5）。

PNG そのものではなく、記録から組み立てた表示内容と HTML を検証する。
ブラウザ無しで「配信した内容と台帳が一致すること」を固定できる境界がここ（§4.3）。
PNG 化は環境に Chromium がある場合だけスモークで確認する。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.render.context import EXIT_RULE, MA_LABELS, build_context
from stockbot.screener.record import load_delivered, records_to_frame, save_delivered

ASOF = pd.Timestamp("2026-08-31")
DELIVERED_ON = pd.Timestamp("2026-09-01")


def make_record(ticker="2801.T", name="キッコーマン", landing_ma="SMA5", close_t=1835.0,
                lp=1790.5, h0_high=1867.5, depth_pct=0.041231593038821956, days=6,
                adv=7009219892.5, unknown=True, earnings_days=np.nan, streak=1,
                sector="食料品", state="反発開始", dist=0.6436456198738544):
    return {
        "delivered_on": DELIVERED_ON, "asof": ASOF, "ticker": ticker, "name": name,
        "landing_ma": landing_ma, "landing_ma_value": 1822.5, "landing_dist_atr": dist,
        "lp": lp, "lp_date": pd.Timestamp("2026-08-28"),
        "h0_high": h0_high, "h0_date": pd.Timestamp("2026-08-21"),
        "close_t": close_t, "atr_t": 49.7167991390535, "state": state,
        "depth_pct": depth_pct, "pullback_days": days,
        "adv_jpy": adv, "sector33": sector, "a4_earnings_unknown": unknown,
        "e1_skipped": True, "earnings_days": earnings_days,
        "streak": streak, "prev_delivered_on": pd.NaT,
    }


def make_summary(n_candidates=1, n_pool=1, e1_skipped=True, fails=None,
                 level="中", score=3, fetch_ok=3663, fetch_total=3786):
    from stockbot.screener.conditions import SELF_CONTAINED_IDS
    return {
        "delivered_on": "2026-09-01", "asof": "2026-08-31",
        "regime_level": level, "regime_score": score,
        "n_evaluated": 1330, "fetch_ok": fetch_ok, "fetch_total": fetch_total,
        "n_pool": n_pool, "n_candidates": n_candidates,
        "earnings_known": 35, "earnings_coverage": 0.0263,
        "e1_skipped": e1_skipped, "e1_threshold": None,
        "fail_counts": fails if fails is not None else {
            cid: n for cid, n in zip(SELF_CONTAINED_IDS,
                                     [192, 12, 0, 8, 498, 303, 334, 329,
                                      1086, 678, 967, 316, 491, 945, 796, 712, 305, 1053])},
        "landing_ma_all": {"SMA5": 514, "SMA25": 277, "SMA75": 219, "SMA200": 252},
        "landing_ma_candidates": {"SMA5": 1, "SMA25": 0, "SMA75": 0, "SMA200": 0},
    }


def html(delivered, summary):
    from stockbot.render.render import render_html
    return render_html(delivered, summary)


class ContextTest(unittest.TestCase):
    def test_card_fields_come_from_the_record(self):
        c = build_context(records_to_frame([make_record()]), make_summary())["cards"][0]
        self.assertEqual(c["ticker"], "2801.T")
        self.assertEqual(c["name"], "キッコーマン")
        self.assertEqual(c["sector33"], "食料品")
        self.assertEqual(c["state"], "反発開始")
        self.assertEqual(c["close"], "1,835.0")
        self.assertEqual(c["landing_ma"], "5日線")
        self.assertEqual(c["landing_dist"], "0.64 ATR")
        self.assertEqual(c["lp"], "1,790.5")
        self.assertEqual(c["h0_high"], "1,867.5")
        self.assertEqual(c["depth"], "4.12%")
        self.assertEqual(c["pullback_days"], "6日")
        self.assertEqual(c["adv"], "70.1億円")
        self.assertEqual(c["exit_rule"], EXIT_RULE)

    def test_gap_percentages(self):
        """撤退ライン・目標は終値からの乖離率を添える。"""
        c = build_context(records_to_frame([make_record()]), make_summary())["cards"][0]
        # (1790.5 - 1835.0) / 1835.0 = -2.42%
        self.assertEqual(c["lp_gap"], "-2.4%")
        # (1867.5 - 1835.0) / 1835.0 = +1.77%
        self.assertEqual(c["h0_gap"], "+1.8%")

    def test_streak_only_from_second_day(self):
        ctx1 = build_context(records_to_frame([make_record(streak=1)]), make_summary())
        ctx2 = build_context(records_to_frame([make_record(streak=3)]), make_summary())
        self.assertIsNone(ctx1["cards"][0]["streak"])
        self.assertEqual(ctx2["cards"][0]["streak"], "連続3日目")

    def test_earnings_unknown_is_flagged(self):
        known = build_context(records_to_frame([make_record(unknown=False, earnings_days=12.0)]),
                              make_summary())["cards"][0]["earnings"]
        self.assertEqual(known, {"text": "決算まで12営業日", "unknown": False})
        unknown = build_context(records_to_frame([make_record()]),
                                make_summary())["cards"][0]["earnings"]
        self.assertTrue(unknown["unknown"])
        self.assertEqual(unknown["text"], "決算日未取得")

    def test_health_block(self):
        h = build_context(records_to_frame([make_record()]), make_summary())["health"]
        self.assertIn("3,663/3,786", h["fetch_text"])
        self.assertIn("96.8%", h["fetch_text"])  # 3663/3786
        self.assertIn("35/1,330", h["earnings_text"])
        self.assertIn("2.6%", h["earnings_text"])
        self.assertIn("決算前でも候補に出ます", h["earnings_note"])

    def test_health_without_fetch_meta(self):
        h = build_context(records_to_frame([make_record()]),
                          make_summary(fetch_ok=None, fetch_total=None))["health"]
        self.assertEqual(h["fetch_text"], "—")

    def test_zero_day_has_no_cards_and_no_e1_note(self):
        ctx = build_context(records_to_frame([]), make_summary(n_candidates=0, n_pool=0))
        self.assertEqual(ctx["cards"], [])
        self.assertIsNone(ctx["e1_note"])
        self.assertEqual([r["cid"] for r in ctx["zero_day"]["rows"]], ["C1", "E3", "C3"])
        self.assertGreater(ctx["zero_day"]["total"], ctx["zero_day"]["n_evaluated"])

    def test_e1_note_when_candidates_exist(self):
        ctx = build_context(records_to_frame([make_record()]), make_summary())
        self.assertIn("未適用", ctx["e1_note"])

    def test_missing_values_do_not_become_nan(self):
        rec = dict(make_record(), landing_ma="", landing_dist_atr=np.nan, adv_jpy=np.nan)
        c = build_context(records_to_frame([rec]), make_summary())["cards"][0]
        self.assertEqual(c["landing_ma"], "—")
        self.assertEqual(c["landing_dist"], "—")
        self.assertEqual(c["adv"], "—")


class HtmlTest(unittest.TestCase):
    def test_all_required_items_appear(self):
        out = html(records_to_frame([make_record()]), make_summary())
        for expected in ("2801.T", "キッコーマン", "食料品", "1,835.0", "5日線", "0.64 ATR",
                         "1,790.5", "-2.4%", "1,867.5", "+1.8%", "4.12%", "6日",
                         "反発開始", "70.1億円", "決算日未取得", EXIT_RULE,
                         "3,663/3,786", "決算前でも候補に出ます",
                         "順位ではありません", "未適用"):
            self.assertIn(expected, out, expected)

    def test_excluded_items_are_absent(self):
        """入れないと決めたものが混ざっていないこと（合成点数・INゾーン・外部指標）。"""
        out = html(records_to_frame([make_record()]), make_summary())
        for banned in ("期待度", "INゾーン", "指値", "SOX", "S&P500", "VI代理",
                       "セクター", "score", "1単元"):
            self.assertNotIn(banned, out, banned)

    def test_two_pages_and_summary_list(self):
        out = html(records_to_frame([make_record(ticker="1111.T", name="テスト")]),
                   make_summary())
        self.assertIn('id="page1"', out)
        self.assertIn('id="page2"', out)
        self.assertIn("本日のまとめ", out)
        self.assertIn("詳細", out)
        self.assertEqual(out.count("1111.T"), 2)   # 1枚目と2枚目に1回ずつ

    def test_zero_day_html(self):
        out = html(records_to_frame([]), make_summary(n_candidates=0, n_pool=0))
        self.assertIn("19条件を全て満たす銘柄がありませんでした", out)
        self.assertIn("延べ件数", out)
        self.assertIn("深さ3〜8%", out)
        # 銘柄カードそのものが出ない（2枚目のフッタには「撤退ライン」の語が残る）
        self.assertNotIn('class="val stop"', out)
        self.assertNotIn("終値から", out)
        self.assertNotIn("未適用", out)

    def test_matches_record_after_csv_roundtrip(self):
        """CSV に書いて読み直しても HTML が変わらない（配信と台帳の一致、§4.3）。"""
        rec = make_record()
        direct = html(records_to_frame([rec]), make_summary())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_delivered(records_to_frame([rec]), Path(tmp), DELIVERED_ON)
            from_csv = html(load_delivered(path), make_summary())
        self.assertEqual(direct, from_csv)

    def test_all_ma_labels_render(self):
        for ma in ("SMA5", "SMA25", "SMA75", "SMA200"):
            out = html(records_to_frame([make_record(landing_ma=ma)]), make_summary())
            self.assertIn(MA_LABELS[ma], out)


class PngSmokeTest(unittest.TestCase):
    """Chromium がある環境でだけ PNG 化を確認する（CI とローカルの両方で落とさない）。"""

    def test_png_generation(self):
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError:
            self.skipTest("playwright 未導入")
        from stockbot.render.render import render_images

        with tempfile.TemporaryDirectory() as tmp:
            try:
                paths = render_images(records_to_frame([make_record()]), make_summary(),
                                      Path(tmp), stem="smoke")
            except Exception as e:
                self.skipTest(f"Chromium を起動できない: {type(e).__name__}")
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 5000)   # 空画像でない


if __name__ == "__main__":
    unittest.main()
