"""配信記録 → 5営業日後の結果 → 台帳、を合成データで通す（docs/SCREENER.md §3）。

DRYRUN と同じ合成日足（data/synthetic.py）を使い、押し目状態の銘柄に対して
記録・結果付け・集計が一通り動くことを確かめる。
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.store import IDX_TICKER
from stockbot.data.synthetic import make_synthetic
from stockbot.features.indicators import atr_wilder, sma
from stockbot.features.pullback import (
    STATE_BOUNCE,
    STATE_BREAK,
    STATE_FORMING,
    pullback_state,
)
from stockbot.features.swings import alternate_swings, detect_raw_swings
from stockbot.screener.record import build_record, records_to_frame, save_delivered
from stockbot.screener.resolver import (
    HORIZON_DAYS,
    load_journal,
    resolve_pending,
    summarize_by_landing_ma,
)

K = 3
SCORABLE = (STATE_FORMING, STATE_BOUNCE, STATE_BREAK)
N_BARS = 400
# 判定日を末尾から HORIZON_DAYS だけ手前に置くと、結果付けに必要な足が既にある
T_OFFSET = HORIZON_DAYS


class ScreenerEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tickers = [f"{1300 + i}.T" for i in range(60)]
        cls.ohlcv = make_synthetic(tickers, n_bars=N_BARS, end=pd.Timestamp("2026-08-28"))
        cls.ohlcv[IDX_TICKER] = next(iter(cls.ohlcv.values())).copy()

        cls.records = []
        for ticker in tickers:
            df = cls.ohlcv[ticker]
            t_pos = len(df) - 1 - T_OFFSET
            high, low, close = df["High"], df["Low"], df["Close"]
            alt = alternate_swings(detect_raw_swings(high, low, K))
            pb = pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                                atr_wilder(high, low, close, 14), alt, t_pos, K)
            if pb["state"] not in SCORABLE:
                continue
            cls.records.append(build_record(ticker, high, low, close, pb, t_pos,
                                            delivered_on=df.index[t_pos + 1]))
        cls.delivered_on = cls.ohlcv[tickers[0]].index[N_BARS - T_OFFSET]

    def test_some_candidates_exist(self):
        self.assertGreater(len(self.records), 0)

    def test_every_record_has_lp_and_h0(self):
        for rec in self.records:
            self.assertGreater(rec["h0_high"], rec["lp"], rec["ticker"])
            self.assertLess(rec["lp_date"], rec["asof"] + pd.Timedelta(days=1))
            self.assertLess(rec["h0_date"], rec["asof"])

    def test_landing_ma_is_recorded(self):
        lines = {rec["landing_ma"] for rec in self.records}
        self.assertTrue(lines <= {"SMA5", "SMA25", "SMA75", "SMA200"})
        self.assertTrue(lines)

    def test_full_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            save_delivered(records_to_frame(self.records), daily, self.delivered_on)
            written = resolve_pending(daily, self.ohlcv, log=lambda *_a: None)
            self.assertEqual(len(written), 1)

            journal = load_journal(daily)
            self.assertEqual(len(journal), len(self.records))
            self.assertIn("success", journal.columns)
            self.assertTrue(journal["success"].notna().all())
            self.assertFalse(journal["censored"].astype(bool).any())
            self.assertTrue((journal["n_bars"] == HORIZON_DAYS).all())

            summary = summarize_by_landing_ma(journal)
            self.assertEqual(int(summary["n"].sum()), len(self.records))
            self.assertEqual(int(summary["n_resolved"].sum()), len(self.records))
            for rate in summary["success_rate"]:
                self.assertTrue(0.0 <= rate <= 1.0)

            # 2 回目の実行は何もしない（確定した記録を作り直さない）
            self.assertEqual(resolve_pending(daily, self.ohlcv, log=lambda *_a: None), [])


if __name__ == "__main__":
    unittest.main()
