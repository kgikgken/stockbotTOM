"""配信記録（docs/SCREENER.md §3.2）と「止まった線」（features.dimensions.landing_ma）。

時点整合（T より後の足を足しても記録が変わらない）を必須テストとして含む。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.features.dimensions import landing_ma, ma_values_at
from stockbot.features.indicators import atr_wilder, sma
from stockbot.features.pullback import pullback_state
from stockbot.features.swings import alternate_swings, detect_raw_swings
from stockbot.screener.record import (
    DELIVERED_COLS,
    build_record,
    list_delivered,
    load_delivered,
    records_to_frame,
    save_delivered,
)

K = 3


H0_POS = 244  # peak 配列の中央（tests/test_pullback.py と同じ合成系列）


def make_series(tail=(217.0, 214.0, 211.0, 209.0)):
    """warmup（単調増加）→ L0のV字ディップ → 上昇 → H0のピーク → 押し目（tail）。

    tests/test_pullback.py の受け入れ系列と同じ作り方。h0 は H0_POS、確定は h0+k。
    """
    warm_end = 180.0
    warmup = np.linspace(100.0, warm_end, 220)
    dip = np.array([180, 178, 176, 174, 176, 178, 180.0]) + (warmup[-1] - 180)
    rise = np.linspace(dip[-1], 220.0, 15)[1:]
    peak = np.array([220, 222, 224, 230, 224, 222, 220.0])
    close = np.concatenate([warmup, dip, rise, peak, np.asarray(tail, dtype=float)])
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    close_s = pd.Series(close, index=idx, dtype=float)
    high_s = close_s + 0.3
    low_s = close_s - 0.3
    open_s = close_s.shift(1).fillna(close_s.iloc[0])
    return open_s, high_s, low_s, close_s


class TestLandingMa(unittest.TestCase):
    def test_nearest_line_and_distance(self):
        r = landing_ma(1000.0, {"SMA5": 1010.0, "SMA25": 1001.0,
                                "SMA75": 950.0, "SMA200": 800.0}, atr_t=20.0)
        self.assertEqual(r["landing_ma"], "SMA25")
        self.assertAlmostEqual(r["landing_ma_value"], 1001.0)
        self.assertAlmostEqual(r["dist_atr"], 1.0 / 20.0)

    def test_nan_lines_are_skipped(self):
        r = landing_ma(1000.0, {"SMA5": np.nan, "SMA25": np.nan,
                                "SMA75": 990.0, "SMA200": np.nan}, atr_t=10.0)
        self.assertEqual(r["landing_ma"], "SMA75")

    def test_no_candidate_or_no_atr_returns_none(self):
        for kwargs in (
            dict(lp_value=1000.0, ma_values={"SMA5": np.nan}, atr_t=10.0),
            dict(lp_value=1000.0, ma_values={"SMA5": 999.0}, atr_t=0.0),
            dict(lp_value=np.nan, ma_values={"SMA5": 999.0}, atr_t=10.0),
        ):
            r = landing_ma(**kwargs)
            self.assertIsNone(r["landing_ma"])
            self.assertTrue(np.isnan(r["dist_atr"]))

    def test_tie_prefers_shorter_line(self):
        # 同じ距離なら LANDING_MA_NAMES の並び（短い線が先）
        r = landing_ma(1000.0, {"SMA5": 990.0, "SMA25": 1010.0}, atr_t=10.0)
        self.assertEqual(r["landing_ma"], "SMA5")

    def test_ma_values_at_matches_sma(self):
        _o, _h, _l, close = make_series()
        pos = 240
        vals = ma_values_at(close, pos)
        for name in ("SMA5", "SMA25", "SMA75", "SMA200"):
            expected = sma(close, int(name[3:])).iloc[pos]
            self.assertAlmostEqual(vals[name], float(expected))

    def test_ma_values_at_is_causal(self):
        """系列を pos で切って計算しても同じ（再計算一致、CLAUDE.md）。"""
        _o, _h, _l, close = make_series()
        pos = 240
        full = ma_values_at(close, pos)
        cut = ma_values_at(close.iloc[: pos + 1], pos)
        for name, value in full.items():
            self.assertAlmostEqual(value, cut[name])


class TestBuildRecord(unittest.TestCase):
    def _record(self, t_pos=None, cut=False):
        _o, high, low, close = make_series()
        t_pos = len(close) - 1 if t_pos is None else t_pos
        if cut:
            high, low, close = (s.iloc[: t_pos + 1] for s in (high, low, close))
        sma5 = sma(close, 5)
        sma200 = sma(close, 200)
        atr14 = atr_wilder(high, low, close, 14)
        alt = alternate_swings(detect_raw_swings(high, low, K))
        pb = pullback_state(high, low, close, sma5, sma200, atr14, alt, t_pos, K)
        return build_record("1234.T", high, low, close, pb, t_pos,
                            delivered_on="2024-12-31", name="テスト銘柄"), pb, close

    def test_record_has_all_columns(self):
        rec, _pb, _close = self._record()
        self.assertEqual(set(rec), set(DELIVERED_COLS))

    def test_record_uses_pullback_values(self):
        rec, pb, close = self._record()
        self.assertEqual(rec["ticker"], "1234.T")
        self.assertEqual(rec["state"], pb["state"])
        self.assertAlmostEqual(rec["lp"], pb["lp_value"])
        self.assertAlmostEqual(rec["h0_high"], pb["h0_high"])
        self.assertAlmostEqual(rec["close_t"], float(close.iloc[-1]))
        self.assertEqual(rec["asof"], pd.Timestamp(close.index[-1]))
        self.assertEqual(rec["delivered_on"], pd.Timestamp("2024-12-31"))

    def test_landing_ma_is_one_of_the_four_lines(self):
        rec, _pb, _close = self._record()
        self.assertIn(rec["landing_ma"], ("SMA5", "SMA25", "SMA75", "SMA200"))
        self.assertTrue(np.isfinite(rec["landing_dist_atr"]))

    def test_record_does_not_look_past_t(self):
        """T より後の足があってもなくても記録は同じ（未来参照の禁止、CLAUDE.md）。"""
        t_pos = 249
        full, _pb, _c = self._record(t_pos=t_pos, cut=False)
        cut, _pb2, _c2 = self._record(t_pos=t_pos, cut=True)
        self.assertEqual(set(full), set(cut))
        for key, value in full.items():
            other = cut[key]
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(isinstance(other, float) and np.isnan(other), key)
            else:
                self.assertEqual(value, other, key)


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip_and_listing(self):
        rec, _pb, _close = TestBuildRecord()._record()
        df = records_to_frame([rec])
        self.assertEqual(list(df.columns), DELIVERED_COLS)
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            path = save_delivered(df, daily, "2024-12-31")
            self.assertEqual(path.name, "delivered_2024-12-31.csv")
            back = load_delivered(path)
            self.assertEqual(len(back), 1)
            self.assertEqual(back["ticker"].iloc[0], "1234.T")
            self.assertEqual(back["landing_ma"].iloc[0], rec["landing_ma"])
            self.assertEqual(back["asof"].iloc[0], rec["asof"])
            self.assertAlmostEqual(back["lp"].iloc[0], rec["lp"])
            files = list_delivered(daily)
            self.assertEqual([d.strftime("%Y-%m-%d") for d, _p in files], ["2024-12-31"])

    def test_existing_file_is_not_overwritten(self):
        rec, _pb, _close = TestBuildRecord()._record()
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            save_delivered(records_to_frame([rec]), daily, "2024-12-31")
            other = dict(rec, ticker="9999.T")
            save_delivered(records_to_frame([other]), daily, "2024-12-31")
            back = load_delivered(daily / "delivered_2024-12-31.csv")
            self.assertEqual(back["ticker"].tolist(), ["1234.T"])

    def test_empty_frame_has_columns(self):
        self.assertEqual(list(records_to_frame([]).columns), DELIVERED_COLS)

    def test_record_without_structure_roundtrips(self):
        """押し目構造が無い（lp/h0 が None）記録も、欠損のまま読み書きできる。"""
        _o, high, low, close = make_series()
        t_pos = len(close) - 1
        no_struct = {"state": "no_structure", "h0": None, "l0": None, "lp": None,
                     "h0_high": np.nan, "lp_value": np.nan, "depth_pct": np.nan, "d": None}
        rec = build_record("5555.T", high, low, close, no_struct, t_pos,
                           delivered_on="2024-12-31")
        self.assertEqual(rec["landing_ma"], "")
        self.assertTrue(pd.isna(rec["lp_date"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_delivered(records_to_frame([rec]), Path(tmp), "2024-12-31")
            back = load_delivered(path)
            self.assertEqual(back["landing_ma"].iloc[0], "")
            self.assertTrue(pd.isna(back["lp"].iloc[0]))
            self.assertTrue(pd.isna(back["lp_date"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
