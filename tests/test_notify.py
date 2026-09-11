"""LINE 配信（docs/PATTERN.md §6）。

要は「配信した内容と記録が食い違わないこと」。本文に出る値がすべて成立の配信記録と
その日のスナップショットの列から来ていることを固定する（SCREENER.md §4.3 と同じ）。

**通常はテキストを送らない。** 画像カード2枚だけを送り、この本文は描画か送信に失敗
した日だけ流れる。`line_send` の caption の扱いもここで固定する（caption を付けると
Worker がテキストを 1 通余分に push する。2026-09-04 に実際に起きた）。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.notify import line_send
from stockbot.notify.line_send import push_image
from stockbot.notify.message import (
    DISCLAIMER,
    FALLBACK_NOTE,
    MAX_TEXT,
    TARGET_NOTE,
    build_message,
)
from stockbot.render.context import WATCH_MAX


def done_row(ticker="5202.T", pattern="inverse_hs", **kw):
    row = {
        "ticker": ticker, "name": "日本板硝子", "sector33": "ガラス・土石製品",
        "pattern": pattern, "close_t": 495.0, "neckline": 493.0, "breakout_pct": 0.41,
        "pattern_low": 478.0, "target": 508.0, "up_pct": 2.6, "down_pct": -3.4,
        "rr": 0.76, "adv_jpy": 5.2e8, "earnings_days": np.nan, "earnings_unknown": True,
    }
    row.update(kw)
    return row


def watch_row(ticker="2331.T", pattern="ascending_triangle", **kw):
    row = {"ticker": ticker, "pattern": pattern, "breakout_pct": -0.07, "watch_streak": 1}
    row.update(kw)
    return row


def summary(**kw):
    out = {"delivered_on": "2026-09-14", "asof": "2026-09-11",
           "n_evaluated": 1277, "n_watch": 196}
    out.update(kw)
    return out


def frame(rows):
    return pd.DataFrame(rows) if rows else None


class MatchesRecordTest(unittest.TestCase):
    """本文の値はすべて記録の列から来る（何も計算し直さない）。"""

    def test_every_required_field_appears(self):
        text = build_message(frame([done_row()]), None, summary())
        for token in ("5202.T", "日本板硝子", "逆三尊", "495.0", "+0.41%",
                      "478.0", "508.0", "0.76", "ガラス・土石製品", "5.2億円"):
            self.assertIn(token, text, token)

    def test_target_is_labelled_as_literature(self):
        """測定目標は文献上の目安だと必ず書く（§2.3）。"""
        self.assertIn(TARGET_NOTE, build_message(frame([done_row()]), None, summary()))
        self.assertIn("文献上の目安", TARGET_NOTE)

    def test_earnings_days_when_known(self):
        row = done_row(earnings_days=5.0, earnings_unknown=False)
        self.assertIn("決算まで5営業日", build_message(frame([row]), None, summary()))

    def test_earnings_unknown_is_stated_not_hidden(self):
        """9 割方これになる。それでも出す（§6.2）。"""
        self.assertIn("決算日未取得", build_message(frame([done_row()]), None, summary()))

    def test_missing_values_render_as_dash_not_nan(self):
        row = done_row(rr=np.nan, target=np.nan, adv_jpy=np.nan)
        text = build_message(frame([row]), None, summary())
        self.assertNotIn("nan", text.lower())
        self.assertIn("—", text)

    def test_disclaimer_is_present(self):
        self.assertIn(DISCLAIMER, build_message(frame([done_row()]), None, summary()))


class ZeroDoneTest(unittest.TestCase):
    """**成立0件の日も配信する**（§6.1）。何も送らないと壊れたのか分からない。"""

    def test_states_that_there_is_none(self):
        text = build_message(None, frame([watch_row()]), summary())
        self.assertIn("本日の成立はありません", text)
        self.assertIn("成立 0件", text)

    def test_watch_is_still_listed(self):
        text = build_message(None, frame([watch_row()]), summary())
        self.assertIn("2331.T", text)
        self.assertIn("上昇三角", text)

    def test_empty_both(self):
        text = build_message(None, None, summary(n_watch=0))
        self.assertIn("本日の成立はありません", text)
        self.assertNotIn("■ 監視", text)


class WatchTest(unittest.TestCase):
    def test_done_ticker_is_dropped_from_watch(self):
        """**同一銘柄が成立していれば監視から外す**（§6.4）。

        5202.T は逆三尊が成立し、上昇三角が未抜けだった（2026-09-10 の実データ）。
        既に抜けた銘柄を「抜けるのを待つ」側に出すと読み手が混乱する。
        """
        done = frame([done_row("5202.T", "inverse_hs")])
        watch = frame([watch_row("5202.T", "ascending_triangle"),
                       watch_row("2331.T", "ascending_triangle")])
        text = build_message(done, watch, summary())
        self.assertIn("■ 監視", text)
        self.assertIn("2331.T", text)
        watch_part = text.split("■ 監視")[1]
        self.assertNotIn("5202.T", watch_part)

    def test_capped_at_watch_max(self):
        watch = frame([watch_row(f"{1000 + i}.T") for i in range(WATCH_MAX + 8)])
        text = build_message(None, watch, summary())
        self.assertIn(f"上位 {WATCH_MAX}件", text)
        self.assertEqual(text.count("［上昇三角］"), WATCH_MAX)

    def test_streak_shown_only_after_the_first_day(self):
        first = build_message(None, frame([watch_row(watch_streak=1)]), summary())
        self.assertNotIn("監視1日目", first)
        later = build_message(None, frame([watch_row(watch_streak=4)]), summary())
        self.assertIn("監視4日目", later)

    def test_broken_streak_value_does_not_crash(self):
        text = build_message(None, frame([watch_row(watch_streak=None)]), summary())
        self.assertIn("2331.T", text)

    def test_full_watch_count_is_from_the_summary_not_the_shown_rows(self):
        """見出しは検出した全件（196）で、1枚目に載せた数ではない。"""
        text = build_message(None, frame([watch_row()]), summary(n_watch=196))
        self.assertIn("監視 196件", text)


class FallbackTextTest(unittest.TestCase):
    """この本文が流れるのは描画か送信に失敗した日だけ（§6）。"""

    def test_fallback_prefix_is_first_line(self):
        text = build_message(frame([done_row()]), None, summary(), fallback=True)
        self.assertEqual(text.splitlines()[0], FALLBACK_NOTE)

    def test_no_prefix_when_not_a_fallback(self):
        text = build_message(frame([done_row()]), None, summary())
        self.assertNotIn(FALLBACK_NOTE, text)


class LengthTest(unittest.TestCase):
    def test_within_limit(self):
        watch = frame([watch_row(f"{1000 + i}.T") for i in range(WATCH_MAX)])
        done = frame([done_row(f"{2000 + i}.T") for i in range(40)])
        text = build_message(done, watch, summary())
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




class _FakeResponse:
    def __init__(self, status_code: int, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class ImageOnlyDeliveryTest(unittest.TestCase):
    """通常配信は画像2枚だけ（docs/SCREENER.md §4.5）。

    Worker の /upload は caption を付けると画像とは別にテキストを 1 通 push する
    （`src/worker.js`）。2026-09-04 はこれで本文と画像の両方が届いていた。
    """

    def test_push_image_sends_no_caption_field_when_empty(self):
        calls = []

        def fake_post(url, files=None, data=None, headers=None, timeout=None):
            calls.append({"url": url, "data": data})
            return _FakeResponse(200)

        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "a.png"
            img.write_bytes(b"x")
            res = push_image(img, url="https://example.test", post=fake_post)
        self.assertTrue(res["sent"])
        self.assertEqual(calls[0]["url"], "https://example.test/upload")
        # caption が空なら multipart に caption を入れない → Worker はテキストを流さない
        self.assertEqual(calls[0]["data"], {})


if __name__ == "__main__":
    unittest.main()
