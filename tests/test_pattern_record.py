"""パターンの配信記録と日次スナップショット（docs/PATTERN.md §3）。

固定するのは 3 つ。**記録するのは成立だけ**、**スナップショットは上書きしてよい**、
**監視の連続日数は (銘柄, パターン) 単位で数える**。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.screener.pattern_record import (
    PATTERN_DELIVERED_COLS,
    build_delivered,
    is_pattern_record,
    latest_pattern_summary,
    list_pattern_summaries,
    load_pattern_delivered,
    pattern_summary_path,
    save_pattern_summary,
    watch_streaks,
)
from stockbot.screener.record import save_delivered


def rows(n=2):
    """step_pattern が作る表のうち、記録に要る列だけを模したもの。"""
    return pd.DataFrame({
        "ticker": [f"100{i}.T" for i in range(n)],
        "name": [f"銘柄{i}" for i in range(n)],
        "sector33": ["機械"] * n,
        "pattern": ["ascending_triangle"] * n,
        "neckline": [100.0 + i for i in range(n)],
        "close_t": [101.0 + i for i in range(n)],
        "breakout_pct": [1.0] * n,
        "pattern_low": [90.0] * n,
        "height": [10.0] * n,
        "target": [110.0] * n,
        "span": [25] * n,
        "l1_date": [pd.Timestamp("2026-08-03")] * n,
        "l1": [90.0] * n,
        "h1_date": [pd.Timestamp("2026-08-17")] * n,
        "h1": [100.0] * n,
        "adv_jpy": [5e8] * n,
        "earnings_days": [np.nan] * n,
        "earnings_unknown": [True] * n,
    })


class BuildDeliveredTest(unittest.TestCase):
    def test_columns_and_dates(self):
        out = build_delivered(rows(), "2026-09-14", "2026-09-11")
        self.assertEqual(list(out.columns), PATTERN_DELIVERED_COLS)
        self.assertEqual(out["delivered_on"].iloc[0], pd.Timestamp("2026-09-14"))
        self.assertEqual(out["asof"].iloc[0], pd.Timestamp("2026-09-11"))

    def test_missing_columns_become_blank_not_an_error(self):
        """検出側に無い列（保ち合い系でない行の pole_pct など）は欠損で埋める。"""
        out = build_delivered(rows(), "2026-09-14", "2026-09-11")
        self.assertTrue(pd.isna(out["pole_pct"].iloc[0]))
        self.assertTrue(pd.isna(out["l3_date"].iloc[0]))
        self.assertEqual(out["l3_date"].dtype.kind, "M")

    def test_does_not_recompute(self):
        """値は検出時のものをそのまま持つ（§4.3）。ここで計算し直さない。"""
        src = rows()
        out = build_delivered(src, "2026-09-14", "2026-09-11")
        for col in ("neckline", "close_t", "target", "pattern_low"):
            self.assertEqual(list(out[col]), list(src[col]))

    def test_round_trip_through_csv(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_delivered(rows(), "2026-09-14", "2026-09-11")
            path, written = save_delivered(out, Path(d), "2026-09-14", "2026-09-11")
            self.assertTrue(written)
            back = load_pattern_delivered(path)
            self.assertEqual(list(back["ticker"]), list(out["ticker"]))
            self.assertEqual(back["l1_date"].iloc[0], pd.Timestamp("2026-08-03"))
            self.assertTrue(bool(back["earnings_unknown"].iloc[0]))

    def test_is_pattern_record_distinguishes_from_pullback(self):
        out = build_delivered(rows(), "2026-09-14", "2026-09-11")
        self.assertTrue(is_pattern_record(out))
        pullback = pd.DataFrame({"ticker": ["1.T"], "lp": [1.0], "h0_high": [2.0]})
        self.assertFalse(is_pattern_record(pullback))


class SnapshotTest(unittest.TestCase):
    """日次スナップショット（§3.2）。**台帳ではないので上書きしてよい。**"""

    def test_overwrites_unlike_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            save_pattern_summary({"n_watch": 1}, Path(d), "2026-09-14", "2026-09-11")
            path = save_pattern_summary({"n_watch": 9}, Path(d), "2026-09-14", "2026-09-11")
            self.assertEqual(json.loads(path.read_text())["n_watch"], 9)

    def test_asof_is_in_the_name(self):
        """同じ配信日に引け前と引け後の 2 回走っても両方残る（2026-09-03 の形）。"""
        with tempfile.TemporaryDirectory() as d:
            save_pattern_summary({"n": 0}, Path(d), "2026-09-14", "2026-09-11")
            save_pattern_summary({"n": 5}, Path(d), "2026-09-14", "2026-09-14")
            files = list_pattern_summaries(Path(d))
            self.assertEqual(len(files), 2)
            # 最新の判定（引け後）を返す
            latest = latest_pattern_summary(Path(d), "2026-09-14")
            self.assertEqual(latest.asof, pd.Timestamp("2026-09-14"))
            self.assertEqual(json.loads(latest.path.read_text())["n"], 5)

    def test_does_not_collide_with_the_preserved_screen_summary(self):
        """19条件の `screen_summary_*.json` は保全対象。別の prefix にしてある。"""
        name = pattern_summary_path(Path("/tmp"), "2026-09-14", "2026-09-11").name
        self.assertTrue(name.startswith("pattern_summary_"))
        self.assertNotIn("screen_summary", name)


class WatchStreakTest(unittest.TestCase):
    """監視の連続日数（§6.2）。今日を 1 日目として数える。"""

    def _write(self, d, delivered_on, watch):
        save_pattern_summary({"watch": watch}, Path(d), delivered_on, delivered_on)

    def test_counts_consecutive_days(self):
        with tempfile.TemporaryDirectory() as d:
            for day in ("2026-09-08", "2026-09-09", "2026-09-10"):
                self._write(d, day, [{"ticker": "1.T", "pattern": "c1"}])
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 4)   # 今日を含めて 4 日目

    def test_first_day_is_one(self):
        with tempfile.TemporaryDirectory() as d:
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 1)

    def test_a_gap_breaks_the_streak(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "2026-09-08", [{"ticker": "1.T", "pattern": "c1"}])
            self._write(d, "2026-09-09", [])                      # 出なかった日
            self._write(d, "2026-09-10", [{"ticker": "1.T", "pattern": "c1"}])
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 2)

    def test_counted_per_ticker_and_pattern_not_per_ticker(self):
        """**1 行 1 パターンなので、行に対応する単位で数える。**

        同じ銘柄が C1 では 3 日目、C2 では 1 日目、ということが起こる。
        銘柄だけで数えると両方に同じ数字が出てしまう。
        """
        with tempfile.TemporaryDirectory() as d:
            for day in ("2026-09-09", "2026-09-10"):
                self._write(d, day, [{"ticker": "1.T", "pattern": "c1"}])
            got = watch_streaks(Path(d), [("1.T", "c1"), ("1.T", "c2")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 3)
            self.assertEqual(got[("1.T", "c2")], 1)

    def test_two_judgements_on_one_day_count_once(self):
        """同じ配信日に判定が 2 つある日は和集合。1 日を 2 日と数えない（§3.2）。"""
        with tempfile.TemporaryDirectory() as d:
            save_pattern_summary({"watch": [{"ticker": "1.T", "pattern": "c1"}]},
                                 Path(d), "2026-09-10", "2026-09-09")
            save_pattern_summary({"watch": [{"ticker": "1.T", "pattern": "c1"}]},
                                 Path(d), "2026-09-10", "2026-09-10")
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 2)

    def test_today_is_not_counted_from_files(self):
        """今日のスナップショットが既にあっても二重に数えない。"""
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "2026-09-11", [{"ticker": "1.T", "pattern": "c1"}])
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 1)

    def test_unreadable_file_is_treated_as_absent(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "2026-09-09", [{"ticker": "1.T", "pattern": "c1"}])
            bad = pattern_summary_path(Path(d), "2026-09-10", "2026-09-10")
            bad.write_text("{ broken", encoding="utf-8")
            got = watch_streaks(Path(d), [("1.T", "c1")], "2026-09-11")
            self.assertEqual(got[("1.T", "c1")], 1)   # 09-10 で途切れる

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(watch_streaks(Path(d), [], "2026-09-11"), {})


if __name__ == "__main__":
    unittest.main()
