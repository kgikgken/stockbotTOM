"""LINE 配信（docs/SCREENER.md §4）。

要は「配信した内容と配信記録が食い違わないこと」。カードに出る数値がすべて
delivered_*.csv の列から来ていることをテストで固定する（§4.3）。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.notify import line_send
from stockbot.notify.message import MA_LABELS, MAX_TEXT, TOP_FAILS, build_message
from stockbot.screener.conditions import CONDITION_IDS, CONDITION_LABELS, SELF_CONTAINED_IDS
from stockbot.screener.record import (
    DELIVERED_COLS,
    load_delivered,
    records_to_frame,
    save_delivered,
)

ASOF = pd.Timestamp("2026-08-31")
DELIVERED_ON = pd.Timestamp("2026-09-01")


def make_record(ticker="2801.T", name="キッコーマン", landing_ma="SMA5",
                close_t=1835.0, lp=1790.5, h0_high=1867.5, depth_pct=0.041231593038821956,
                days=6, adv=7009219892.5, unknown=True, earnings_days=np.nan,
                e1_skipped=True, dist=0.6436456198738544):
    return {
        "delivered_on": DELIVERED_ON, "asof": ASOF, "ticker": ticker, "name": name,
        "landing_ma": landing_ma, "landing_ma_value": 1822.5, "landing_dist_atr": dist,
        "lp": lp, "lp_date": pd.Timestamp("2026-08-28"),
        "h0_high": h0_high, "h0_date": pd.Timestamp("2026-08-21"),
        "close_t": close_t, "atr_t": 49.7167991390535, "state": "反発開始",
        "depth_pct": depth_pct, "pullback_days": days,
        "adv_jpy": adv, "sector33": "食料品", "a4_earnings_unknown": unknown,
        "e1_skipped": e1_skipped, "earnings_days": earnings_days,
    }


def make_summary(n_candidates=1, n_pool=1, e1_skipped=True, fails=None, level="中", score=3):
    return {
        "delivered_on": "2026-09-01", "asof": "2026-08-31",
        "regime_level": level, "regime_score": score,
        "n_evaluated": 1330, "n_pool": n_pool, "n_candidates": n_candidates,
        "e1_skipped": e1_skipped, "e1_threshold": None,
        "fail_counts": fails if fails is not None else {
            cid: n for cid, n in zip(SELF_CONTAINED_IDS,
                                     [192, 12, 0, 8, 498, 303, 334, 329,
                                      1086, 678, 967, 316, 491, 945, 796, 712, 305, 1053])},
        "landing_ma_all": {"SMA5": 514, "SMA25": 277, "SMA75": 219, "SMA200": 252},
        "landing_ma_candidates": {"SMA5": 1, "SMA25": 0, "SMA75": 0, "SMA200": 0},
    }


def frame(records):
    return records_to_frame(records)


class MatchesDeliveredTest(unittest.TestCase):
    """§4.3: カードの値はすべて配信記録の列から来る。"""

    def test_every_required_field_appears(self):
        rec = make_record()
        text = build_message(frame([rec]), make_summary())
        self.assertIn("2801.T", text)                 # 銘柄コード
        self.assertIn("キッコーマン", text)             # 銘柄名
        self.assertIn("1,835.0円", text)              # 終値
        self.assertIn("5日線", text)                   # 止まった線
        self.assertIn("0.64 ATR", text)               # 距離
        self.assertIn("1,790.5円", text)              # 押し安値
        self.assertIn("1,867.5円", text)              # 直近高値
        self.assertIn("4.12%", text)                  # 深さ
        self.assertIn("押し目6日", text)                # 押し目日数
        self.assertIn("反発開始", text)                 # 状態
        self.assertIn("70.1億円", text)                # 20日平均売買代金
        self.assertIn("決算日未取得", text)             # 決算までの日数（未取得）

    def test_labels_for_stop_line_and_target(self):
        text = build_message(frame([make_record()]), make_summary())
        self.assertIn("撤退ライン（押し安値）", text)
        self.assertIn("目標の目安（直近高値）", text)

    def test_earnings_days_when_known(self):
        rec = make_record(unknown=False, earnings_days=12.0)
        text = build_message(frame([rec]), make_summary())
        self.assertIn("決算まで12営業日", text)
        self.assertNotIn("決算日未取得", text)

    def test_values_survive_csv_roundtrip(self):
        """CSV に書いて読み直しても本文が変わらない（配信と台帳の一致）。"""
        rec = make_record()
        direct = build_message(frame([rec]), make_summary())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_delivered(frame([rec]), Path(tmp), DELIVERED_ON)
            from_csv = build_message(load_delivered(path), make_summary())
        self.assertEqual(direct, from_csv)

    def test_all_ma_labels_covered(self):
        for ma in ("SMA5", "SMA25", "SMA75", "SMA200"):
            text = build_message(frame([make_record(landing_ma=ma)]), make_summary())
            self.assertIn(MA_LABELS[ma], text)

    def test_missing_values_render_as_dash_not_nan(self):
        rec = dict(make_record(), landing_ma="", landing_dist_atr=np.nan, adv_jpy=np.nan)
        text = build_message(frame([rec]), make_summary())
        self.assertNotIn("nan", text.lower())
        self.assertIn("—", text)


class HeaderTest(unittest.TestCase):
    def test_gauge_and_count(self):
        text = build_message(frame([make_record()]), make_summary())
        self.assertIn("地合い 中（3/6）", text)
        self.assertIn("候補 1件", text)
        self.assertIn("判定 2026-08-31 の引け", text)

    def test_not_a_ranking(self):
        text = build_message(frame([make_record()]), make_summary())
        self.assertIn("順位ではありません", text)
        self.assertIn("売買代金の降順", text)

    def test_e1_skipped_line(self):
        text = build_message(frame([make_record()]), make_summary(e1_skipped=True, n_pool=1))
        self.assertIn("E1", text)
        self.assertIn("未適用", text)

    def test_no_e1_line_when_applied(self):
        text = build_message(frame([make_record(e1_skipped=False)]),
                             make_summary(e1_skipped=False, n_pool=25))
        self.assertNotIn("未適用", text)

    def test_unknown_gauge(self):
        text = build_message(frame([make_record()]), make_summary(level=None, score=None))
        self.assertIn("地合い 不明", text)


class ZeroCandidateTest(unittest.TestCase):
    """§4.2: 候補0件の日も配信し、落ちた条件を上位3つ添える。"""

    def test_zero_day_message(self):
        text = build_message(frame([]), make_summary(n_candidates=0, n_pool=0))
        self.assertIn("候補 0件", text)
        self.assertIn("19条件を全て満たす銘柄がありませんでした", text)

    def test_top_three_failed_conditions(self):
        text = build_message(frame([]), make_summary(n_candidates=0))
        # 上位3つ: C1(1086) E3(1053) C3(967)
        for cid in ("C1", "E3", "C3"):
            self.assertIn(cid, text)
            self.assertIn(CONDITION_LABELS[cid], text)
        self.assertNotIn("A3", text)   # 0件の条件は上位に入らない
        self.assertEqual(text.count(" … "), TOP_FAILS)

    def test_zero_day_with_none_delivered(self):
        text = build_message(None, make_summary(n_candidates=0))
        self.assertIn("候補 0件", text)

    def test_zero_day_without_fail_counts(self):
        s = make_summary(n_candidates=0)
        s["fail_counts"] = {}
        text = build_message(frame([]), s)
        self.assertIn("ありませんでした", text)

    def test_condition_labels_cover_all_ids(self):
        self.assertEqual(set(CONDITION_LABELS), set(CONDITION_IDS))


class LengthTest(unittest.TestCase):
    def test_within_limit_and_reports_omission(self):
        records = [make_record(ticker=f"{1000+i}.T", name="テスト銘柄" * 3)
                   for i in range(120)]
        text = build_message(frame(records), make_summary(n_candidates=120))
        self.assertLessEqual(len(text), MAX_TEXT)
        self.assertIn("省略しました", text)
        self.assertIn("ご自身で", text)   # 免責は必ず残る

    def test_normal_size_has_no_omission_note(self):
        records = [make_record(ticker=f"{1000+i}.T") for i in range(5)]
        text = build_message(frame(records), make_summary(n_candidates=5))
        self.assertNotIn("省略しました", text)
        self.assertLessEqual(len(text), MAX_TEXT)


class PushTest(unittest.TestCase):
    class _Resp:
        def __init__(self, code): self.status_code, self.text = code, "{}"

    def test_skips_without_url(self):
        r = line_send.push_text("x", url=None, token=None, post=lambda *a, **k: None)
        self.assertFalse(r["sent"])
        self.assertIn("WORKER_URL", r["reason"])

    def test_skips_empty_text(self):
        r = line_send.push_text("", url="https://example.invalid", post=lambda *a, **k: None)
        self.assertFalse(r["sent"])

    def test_posts_json_and_auth_header(self):
        seen = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.update(url=url, json=json, headers=headers, timeout=timeout)
            return self._Resp(200)

        r = line_send.push_text("本文", url="https://example.invalid/", token="tok",
                                post=fake_post)
        self.assertTrue(r["sent"])
        self.assertEqual(seen["json"], {"text": "本文"})
        self.assertEqual(seen["headers"]["Authorization"], "Bearer tok")

    def test_no_auth_header_without_token(self):
        seen = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.update(headers=headers)
            return self._Resp(200)

        line_send.push_text("本文", url="https://example.invalid/", token="", post=fake_post)
        self.assertNotIn("Authorization", seen["headers"])

    def test_failure_is_reported_not_swallowed(self):
        r = line_send.push_text("本文", url="https://example.invalid/", token=None,
                                post=lambda *a, **k: self._Resp(502))
        self.assertFalse(r["sent"])
        self.assertEqual(r["status"], 502)
        self.assertIn("502", r["reason"])


if __name__ == "__main__":
    unittest.main()
