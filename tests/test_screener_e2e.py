"""スクリーニング → 配信記録 → 5営業日後の結果 → 台帳（docs/SCREENER.md §2・§3）。

前半は DRYRUN と同じ合成日足（data/synthetic.py）で記録・結果付け・集計が動くこと、
後半は 19 条件を通る系列を並べて「条件判定から台帳まで」が一本で通ることを確かめる。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
from stockbot.screener import screen
from stockbot.screener.record import (
    DELIVERED_COLS,
    build_record,
    records_to_frame,
    save_delivered,
)
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
        cls.asof = cls.records[0]["asof"] if cls.records else cls.delivered_on

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
            save_delivered(records_to_frame(self.records), daily, self.delivered_on, self.asof)
            written = resolve_pending(daily, self.ohlcv, log=lambda *_a: None)
            self.assertEqual(len(written), 1)

            journal = load_journal(daily)
            self.assertEqual(len(journal), len(self.records))
            self.assertIn("success", journal.columns)
            self.assertTrue(journal["success"].notna().all())
            self.assertFalse(journal["censored"].astype(bool).any())
            self.assertTrue((journal["n_bars"] == HORIZON_DAYS).all())

            summary = summarize_by_landing_ma(journal)
            self.assertEqual(summary["landing_ma"].tolist()[:4],
                             ["SMA5", "SMA25", "SMA75", "SMA200"])
            self.assertEqual(int(summary["n"].sum()), len(self.records))
            self.assertEqual(int(summary["n_resolved"].sum()), len(self.records))
            for rate in summary["success_rate"].dropna():
                self.assertTrue(0.0 <= rate <= 1.0)

            # 2 回目の実行は何もしない（確定した記録を作り直さない）
            self.assertEqual(resolve_pending(daily, self.ohlcv, log=lambda *_a: None), [])


if __name__ == "__main__":
    unittest.main()


class ScreenToJournalTest(unittest.TestCase):
    """19 条件を通る系列を 12 銘柄ぶん並べ、条件判定 → 配信記録 → 結果 → 台帳を通す。"""

    N_TICKERS = 12
    SECTORS = {0: "電気機器", 1: "電気機器", 2: "電気機器", 3: "電気機器", 4: "電気機器"}

    @classmethod
    def setUpClass(cls):
        from .test_screener_conditions import base_closes, make_frame, make_index

        cls.tickers = [f"{1400 + i}.T" for i in range(cls.N_TICKERS)]
        # 判定日 T の時点のフレーム（末尾が T）と、5 営業日ぶん足を伸ばしたフレーム
        cls.at_t, cls.after = {}, {}
        closes = base_closes()
        future = [645.0, 650.0, 656.0, 661.0, 665.0]     # 直近高値(660.3)を更新して終わる
        for i, ticker in enumerate(cls.tickers):
            cls.at_t[ticker] = make_frame(closes, volume=(cls.N_TICKERS - i) * 1e6)
            cls.after[ticker] = make_frame(np.concatenate([closes, future]),
                                           volume=(cls.N_TICKERS - i) * 1e6)
        cls.idx_close = make_index(len(closes))
        cls.after[IDX_TICKER] = make_frame(np.concatenate([closes, future]))
        cls.sector_by_ticker = {t: cls.SECTORS.get(i, f"業種{i}")
                                for i, t in enumerate(cls.tickers)}
        cls.asof = cls.at_t[cls.tickers[0]].index[-1]
        cls.delivered_on = cls.after[cls.tickers[0]].index[len(closes)]

    def _screen(self):
        df = screen.evaluate_universe(self.at_t, self.tickers, self.idx_close, K,
                                      log=lambda *_a: None)
        df, meta = screen.apply_e1(df, log=lambda *_a: None)
        return screen.select_candidates(df, self.sector_by_ticker), meta

    def test_all_tickers_pass_the_eighteen(self):
        df = screen.evaluate_universe(self.at_t, self.tickers, self.idx_close, K,
                                      log=lambda *_a: None)
        df, meta = screen.apply_e1(df, log=lambda *_a: None)
        self.assertEqual(meta["e1_pool_n"], self.N_TICKERS)
        self.assertFalse(meta["e1_skipped"])          # 母集団 12 件 >= 10
        # rs60 が全銘柄同値なので「上位10%より大きい」銘柄は無く、全て通過する
        self.assertEqual(int(df["passes"].sum()), self.N_TICKERS)

    def test_sector_cap_and_turnover_order(self):
        candidates, _meta = self._screen()
        self.assertEqual(candidates["ticker"].tolist()[0], self.tickers[0])   # 売買代金 最大
        self.assertTrue((candidates["adv_jpy"].diff().dropna() <= 0).all())
        counts = candidates["sector33"].value_counts().to_dict()
        self.assertEqual(counts["電気機器"], screen.SECTOR_CAP)
        self.assertEqual(len(candidates), self.N_TICKERS - 5 + screen.SECTOR_CAP)

    def test_screen_to_journal(self):
        candidates, meta = self._screen()
        records = []
        for _i, cand in candidates.iterrows():
            ticker = str(cand["ticker"])
            df = self.at_t[ticker]
            high, low, close = df["High"], df["Low"], df["Close"]
            t_pos = len(df) - 1
            alt = alternate_swings(detect_raw_swings(high, low, K))
            pb = pullback_state(high, low, close, sma(close, 5), sma(close, 200),
                                atr_wilder(high, low, close, 14), alt, t_pos, K)
            records.append(build_record(
                ticker, high, low, close, pb, t_pos, self.delivered_on,
                extra={"adv_jpy": float(cand["adv_jpy"]), "sector33": str(cand["sector33"]),
                       "a4_earnings_unknown": bool(cand["a4_earnings_unknown"]),
                       "e1_skipped": meta["e1_skipped"]}))

        with tempfile.TemporaryDirectory() as tmp:
            daily = Path(tmp)
            delivered = records_to_frame(records)
            self.assertEqual(list(delivered.columns), DELIVERED_COLS)
            save_delivered(delivered, daily, self.delivered_on, self.asof)
            written = resolve_pending(daily, self.after, log=lambda *_a: None)
            self.assertEqual(len(written), 1)

            journal = load_journal(daily)
            self.assertEqual(len(journal), len(candidates))
            self.assertEqual(journal["asof"].nunique(), 1)
            self.assertEqual(journal["asof"].iloc[0], self.asof)
            # 決算予定日を渡していないので A4 は「決算日未取得」扱い（カードに明記する）
            self.assertTrue(journal["a4_earnings_unknown"].astype(bool).all())
            self.assertFalse(journal["e1_skipped"].astype(bool).any())
            # 未来の 5 本で直近高値を更新し、押し安値は割っていない
            self.assertTrue(journal["success"].astype(bool).all())
            self.assertFalse(journal["broke_lp"].astype(bool).any())

            summary = summarize_by_landing_ma(journal)
            self.assertEqual(int(summary["n"].sum()), len(candidates))
            self.assertTrue((summary["success_rate"].dropna() == 1.0).all())
            # 候補が出ていない線は n=0 の行として残る（消えない）
            self.assertEqual(len(summary), 4)
