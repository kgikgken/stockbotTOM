"""パターンの結果付け（docs/PATTERN.md §3.4 / §4 Q-3 の回答）。

固定するのは 4 つ。**評価窓 20 本**、**成功＝撤退を割る前に目標到達（同日は失敗）**、
**MFE/MAE を ATR 単位で出す**、**押し目型の記録とは混ざらない**。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.screener.pattern_resolver import (
    PATTERN_HORIZON_DAYS,
    PATTERN_OUTCOME_COLS,
    load_outcome,
    outcome_path,
    resolve_delivered,
    resolve_row,
    save_outcome,
)

ASOF = pd.Timestamp("2026-09-11")


def bars(closes, highs=None, lows=None, opens=None, start="2026-09-11"):
    """T を先頭にした日足。T+1 以降が評価窓になる。"""
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": np.asarray(opens if opens is not None else c, dtype=float),
        "High": np.asarray(highs if highs is not None else c, dtype=float),
        "Low": np.asarray(lows if lows is not None else c, dtype=float),
        "Close": c,
    }, index=idx)


def record(**kw):
    row = {"delivered_on": pd.Timestamp("2026-09-14"), "asof": ASOF, "ticker": "1.T",
           "pattern": "inverse_hs", "close_t": 100.0, "atr_t": 5.0,
           "pattern_low": 90.0, "target": 110.0,
           "up_pct": 10.0, "down_pct": -10.0, "rr": 1.0, "breakeven_win_rate": 0.5}
    row.update(kw)
    return pd.Series(row)


class HorizonTest(unittest.TestCase):
    def test_window_is_twenty_business_days(self):
        """**裁量値。** 5 本では目標まで届かず記録にならない（§4 Q-3）。"""
        self.assertEqual(PATTERN_HORIZON_DAYS, 20)

    def test_uses_exactly_twenty_bars_after_t(self):
        df = bars([100.0] * 30)
        out = resolve_row(record(), df)
        self.assertEqual(out["n_bars"], 20)
        self.assertFalse(out["censored"])

    def test_short_history_is_censored(self):
        out = resolve_row(record(), bars([100.0] * 6))
        self.assertEqual(out["n_bars"], 5)
        self.assertTrue(out["censored"])

    def test_bars_past_the_window_are_ignored(self):
        """21 本目以降で目標に届いても「届いた」にしない。"""
        highs = [100.0] * 21 + [999.0] * 5
        out = resolve_row(record(), bars([100.0] * 26, highs=highs))
        self.assertFalse(bool(out["reached_target"]))


class SuccessTest(unittest.TestCase):
    """成功＝**撤退を割る前に**目標到達。同じ日に両方なら失敗（保守的）。"""

    def test_target_only(self):
        out = resolve_row(record(), bars([100.0] * 21, highs=[100.0] * 5 + [111.0] * 16))
        self.assertTrue(bool(out["reached_target"]))
        self.assertFalse(bool(out["broke_stop"]))
        self.assertTrue(bool(out["success"]))
        self.assertEqual(out["reached_target_day"], 5)

    def test_stop_only(self):
        out = resolve_row(record(), bars([100.0] * 21, lows=[100.0] * 3 + [89.0] * 18))
        self.assertTrue(bool(out["broke_stop"]))
        self.assertFalse(bool(out["reached_target"]))
        self.assertFalse(bool(out["success"]))

    def test_target_before_stop_is_a_success(self):
        highs = [100.0] * 3 + [111.0] * 18
        lows = [100.0] * 9 + [89.0] * 12
        out = resolve_row(record(), bars([100.0] * 21, highs=highs, lows=lows))
        self.assertLess(out["reached_target_day"], out["broke_stop_day"])
        self.assertTrue(bool(out["success"]))

    def test_stop_before_target_is_a_failure(self):
        highs = [100.0] * 9 + [111.0] * 12
        lows = [100.0] * 3 + [89.0] * 18
        out = resolve_row(record(), bars([100.0] * 21, highs=highs, lows=lows))
        self.assertFalse(bool(out["success"]))

    def test_same_day_is_a_failure(self):
        """順序が分からないので失敗側に倒す（押し目型の labels.py と同じ）。"""
        highs = [100.0] * 4 + [111.0] * 17
        lows = [100.0] * 4 + [89.0] * 17
        out = resolve_row(record(), bars([100.0] * 21, highs=highs, lows=lows))
        self.assertEqual(out["reached_target_day"], out["broke_stop_day"])
        self.assertFalse(bool(out["success"]))

    def test_boundaries_are_strict(self):
        """`Low < 撤退` と `High > 目標`。ちょうど同値では成立させない。"""
        touch = resolve_row(record(), bars([100.0] * 21, highs=[110.0] * 21,
                                           lows=[90.0] * 21))
        self.assertFalse(bool(touch["reached_target"]))
        self.assertFalse(bool(touch["broke_stop"]))


class MfeMaeTest(unittest.TestCase):
    """**目標に届かなくても、どこまで伸びたかを残す**（設計責任者の指摘）。"""

    def test_measured_from_the_entry_open(self):
        df = bars([100.0] * 21, highs=[104.0] * 21, lows=[97.0] * 21,
                  opens=[100.0] + [101.0] * 20)
        out = resolve_row(record(), df)
        self.assertEqual(out["entry_open"], 101.0)
        self.assertAlmostEqual(out["mfe"], 3.0)      # 104 − 101
        self.assertAlmostEqual(out["mae"], -4.0)     # 97 − 101

    def test_atr_units(self):
        df = bars([100.0] * 21, highs=[110.0] * 21, lows=[95.0] * 21)
        out = resolve_row(record(atr_t=5.0), df)
        self.assertAlmostEqual(out["mfe_atr"], 2.0)   # +10 / 5
        self.assertAlmostEqual(out["mae_atr"], -1.0)  # −5 / 5

    def test_missing_atr_leaves_only_the_yen_values(self):
        df = bars([100.0] * 21, highs=[110.0] * 21, lows=[95.0] * 21)
        out = resolve_row(record(atr_t=np.nan), df)
        self.assertTrue(np.isfinite(out["mfe"]))
        self.assertTrue(pd.isna(out["mfe_atr"]))

    def test_recorded_even_when_the_target_is_missed(self):
        df = bars([100.0] * 21, highs=[105.0] * 21, lows=[99.0] * 21)
        out = resolve_row(record(), df)
        self.assertFalse(bool(out["reached_target"]))
        self.assertAlmostEqual(out["mfe_atr"], 1.0)


class SchemaTest(unittest.TestCase):
    def test_no_recovered_sma5(self):
        """押し目型の「5日線回復」に対応する概念がパターンには無い。"""
        self.assertNotIn("recovered_sma5", PATTERN_OUTCOME_COLS)
        self.assertNotIn("broke_lp", PATTERN_OUTCOME_COLS)
        self.assertNotIn("reached_h0", PATTERN_OUTCOME_COLS)

    def test_pattern_is_part_of_the_key(self):
        """1 銘柄が複数パターンで成立しうるので、銘柄だけでは行を特定できない。"""
        self.assertIn("pattern", PATTERN_OUTCOME_COLS)

    def test_entry_distances_come_from_the_record(self):
        """配信時点の距離は記録の値をそのまま持つ（結果ではない）。"""
        out = resolve_row(record(rr=0.76, up_pct=2.6), bars([100.0] * 21))
        self.assertAlmostEqual(out["rr"], 0.76)
        self.assertAlmostEqual(out["up_pct"], 2.6)

    def test_separate_prefix_from_the_pullback_outcome(self):
        name = outcome_path(Path("/tmp"), "2026-09-14", "2026-09-11").name
        self.assertTrue(name.startswith("pattern_outcome_"))

    def test_round_trip(self):
        df = bars([100.0] * 21, highs=[111.0] * 21)
        out = resolve_delivered(pd.DataFrame([record()]), {"1.T": df},
                                resolved_on=pd.Timestamp("2026-10-09"))
        self.assertEqual(list(out.columns), PATTERN_OUTCOME_COLS)
        with tempfile.TemporaryDirectory() as d:
            path = save_outcome(out, Path(d), "2026-09-14", "2026-09-11")
            back = load_outcome(path)
            self.assertTrue(bool(back["success"].iloc[0]))
            self.assertEqual(back["pattern"].iloc[0], "inverse_hs")


class MissingDataTest(unittest.TestCase):
    def test_no_ohlcv_is_censored_not_an_error(self):
        out = resolve_row(record(), None)
        self.assertEqual(out["n_bars"], 0)
        self.assertTrue(out["censored"])
        self.assertTrue(pd.isna(out["success"]))

    def test_asof_not_in_the_index(self):
        df = bars([100.0] * 21, start="2026-10-01")
        out = resolve_row(record(), df)
        self.assertEqual(out["n_bars"], 0)

    def test_missing_stop_and_target_leave_the_verdict_unknown(self):
        out = resolve_row(record(pattern_low=np.nan, target=np.nan),
                          bars([100.0] * 21))
        self.assertTrue(pd.isna(out["success"]))


class ResolvePendingTest(unittest.TestCase):
    """`cli resolve` の経路（docs/PATTERN.md §3.4）。

    **押し目型の記録と混ざらないこと**と、**20 営業日が経過するまで持ち越すこと**。
    """

    def _setup(self, tmp, n_bars):
        from stockbot.screener.pattern_record import build_delivered
        from stockbot.screener.record import save_delivered

        df = bars([100.0] * n_bars, highs=[100.0] * 3 + [111.0] * (n_bars - 3))
        rows = pd.DataFrame([{
            "ticker": "1.T", "name": "あ", "sector33": "機械", "pattern": "inverse_hs",
            "close_t": 100.0, "atr_t": 5.0, "neckline": 99.0, "breakout_pct": 1.0,
            "pattern_low": 90.0, "height": 10.0, "target": 110.0,
            "up_pct": 10.0, "down_pct": -10.0, "rr": 1.0, "breakeven_win_rate": 0.5,
            "span": 30, "adv_jpy": 5e8, "earnings_unknown": True,
        }])
        ledger = build_delivered(rows, "2026-09-14", ASOF)
        save_delivered(ledger, Path(tmp), "2026-09-14", ASOF)
        return {"1.T": df}

    def _run(self, tmp, ohlcv):
        from stockbot.screener.resolver import resolve_pending

        lines = []
        written = resolve_pending(Path(tmp), ohlcv, log=lines.append)
        return written, "\n".join(lines)

    def test_resolves_a_pattern_record_with_the_20_bar_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            ohlcv = self._setup(tmp, 26)
            written, text = self._run(tmp, ohlcv)
            self.assertEqual(len(written), 1)
            self.assertTrue(written[0].name.startswith("pattern_outcome_"))
            out = load_outcome(written[0])
            self.assertEqual(int(out["horizon_days"].iloc[0]), PATTERN_HORIZON_DAYS)
            self.assertTrue(bool(out["success"].iloc[0]))
            self.assertIn("成功 1", text)

    def test_holds_over_until_twenty_bars_have_passed(self):
        """**5 本ではなく 20 本**が経過するまで付けない。"""
        with tempfile.TemporaryDirectory() as tmp:
            ohlcv = self._setup(tmp, 10)      # T+9 までしか無い
            written, text = self._run(tmp, ohlcv)
            self.assertEqual(written, [])
            self.assertIn("20営業日", text)

    def test_does_not_redo_an_existing_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            ohlcv = self._setup(tmp, 26)
            self._run(tmp, ohlcv)
            written, _text = self._run(tmp, ohlcv)
            self.assertEqual(written, [])     # 確定した記録は作り直さない

    def test_pullback_records_still_use_their_own_resolver(self):
        """押し目型 15 件は従来の定義で付け続ける（列を混ぜない）。"""
        from stockbot.screener.resolver import OUTCOME_COLS

        self.assertIn("broke_lp", OUTCOME_COLS)
        self.assertIn("recovered_sma5", OUTCOME_COLS)
        self.assertNotIn("broke_stop", OUTCOME_COLS)
        self.assertNotIn("mfe_atr", OUTCOME_COLS)


if __name__ == "__main__":
    unittest.main()
