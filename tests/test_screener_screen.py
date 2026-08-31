"""日次のスクリーニング（docs/SCREENER.md §2.4・§2.5）。

E1（当日候補内での上位10%除外）と、母集団10件未満のスキップ、売買代金降順・
同一33業種3件までの絞り込みを確かめる。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.screener.conditions import DIAGNOSTIC_COLS, SELF_CONTAINED_IDS
from stockbot.screener.screen import (
    E1_MIN_POOL,
    SCREEN_COLS,
    SECTOR_CAP,
    apply_e1,
    build_summary,
    evaluate_universe,
    fail_counts,
    format_counts,
    landing_ma_breakdown,
    save_summary,
    select_candidates,
    summary_path,
)

QUIET = lambda *_a, **_k: None  # noqa: E731


def make_rows(n, rs_values=None, all_pass=True, adv=None):
    """18 条件の判定結果を直接組んだ行（条件判定そのものは test_screener_conditions）。"""
    rows = []
    for i in range(n):
        row = {"ticker": f"{1000 + i}.T", "date": pd.Timestamp("2026-08-28")}
        for cid in SELF_CONTAINED_IDS:
            row[cid] = True if all_pass else (cid != "C1")
        row["E1"] = pd.NA
        row["passes"] = False
        for col in DIAGNOSTIC_COLS:
            row[col] = np.nan
        row["landing_ma"] = "SMA25"
        row["a4_earnings_unknown"] = False
        row["rs60"] = float(i) / n if rs_values is None else rs_values[i]
        row["adv_jpy"] = float(n - i) * 1e8 if adv is None else adv[i]
        row["state"] = "反発開始"
        rows.append(row)
    df = pd.DataFrame(rows, columns=SCREEN_COLS)
    for cid in SELF_CONTAINED_IDS:
        df[cid] = df[cid].astype("boolean")
    df["E1"] = pd.array([pd.NA] * len(df), dtype="boolean")
    return df


class ApplyE1Test(unittest.TestCase):
    def test_drops_top_decile_of_rs60(self):
        df = make_rows(20)   # rs60 = 0.00, 0.05, ..., 0.95
        out, meta = apply_e1(df, log=QUIET)
        self.assertFalse(meta["e1_skipped"])
        self.assertEqual(meta["e1_pool_n"], 20)
        dropped = out[~out["passes"]]["ticker"].tolist()
        self.assertEqual(dropped, ["1018.T", "1019.T"])   # rs60 が上位 10%
        self.assertEqual(int(out["passes"].sum()), 18)

    def test_skips_when_pool_too_small(self):
        df = make_rows(E1_MIN_POOL - 1)
        out, meta = apply_e1(df, log=QUIET)
        self.assertTrue(meta["e1_skipped"])
        self.assertEqual(meta["e1_pool_n"], E1_MIN_POOL - 1)
        self.assertTrue(np.isnan(meta["e1_threshold"]))
        self.assertTrue(out["passes"].all())        # 全て通過（E1 を掛けない）
        self.assertTrue(out["E1"].all())

    def test_boundary_pool_size_applies_e1(self):
        out, meta = apply_e1(make_rows(E1_MIN_POOL), log=QUIET)
        self.assertFalse(meta["e1_skipped"])
        self.assertLess(int(out["passes"].sum()), E1_MIN_POOL)

    def test_e1_pool_is_only_the_eighteen_passers(self):
        """18 条件で落ちた行は母集団に入らず、E1 は未判定（NA）のまま。"""
        good, bad = make_rows(12), make_rows(8, all_pass=False)
        bad["ticker"] = [f"{2000 + i}.T" for i in range(8)]
        df = pd.concat([good, bad], ignore_index=True)
        out, meta = apply_e1(df, log=QUIET)
        self.assertEqual(meta["e1_pool_n"], 12)
        self.assertTrue(out.loc[out["ticker"].str.startswith("2"), "E1"].isna().all())
        self.assertFalse(out.loc[out["ticker"].str.startswith("2"), "passes"].any())

    def test_missing_rs60_fails_e1(self):
        rs = [0.1] * 19 + [np.nan]
        out, meta = apply_e1(make_rows(20, rs_values=rs), log=QUIET)
        self.assertFalse(meta["e1_skipped"])
        self.assertFalse(bool(out.loc[out["ticker"] == "1019.T", "passes"].iloc[0]))

    def test_empty_input(self):
        out, meta = apply_e1(make_rows(0), log=QUIET)
        self.assertEqual(len(out), 0)
        self.assertTrue(meta["e1_skipped"])


class SelectCandidatesTest(unittest.TestCase):
    def _passed(self, n, adv=None):
        df = make_rows(n, rs_values=[0.1] * n, adv=adv)
        out, _meta = apply_e1(df, log=QUIET)
        return out

    def test_sorted_by_turnover_descending(self):
        out = self._passed(12)
        sel = select_candidates(out, {})
        self.assertEqual(sel["ticker"].tolist()[:3], ["1000.T", "1001.T", "1002.T"])
        self.assertTrue((sel["adv_jpy"].diff().dropna() <= 0).all())

    def test_ties_broken_by_ticker(self):
        out = self._passed(12, adv=[5e8] * 12)
        sel = select_candidates(out, {})
        self.assertEqual(sel["ticker"].tolist(), sorted(sel["ticker"].tolist()))

    def test_sector_cap(self):
        out = self._passed(12)
        sectors = {f"{1000 + i}.T": ("電気機器" if i < 8 else "銀行業") for i in range(12)}
        sel = select_candidates(out, sectors)
        counts = sel["sector33"].value_counts().to_dict()
        self.assertEqual(counts.get("電気機器"), SECTOR_CAP)
        self.assertLessEqual(counts.get("銀行業", 0), SECTOR_CAP)
        # 上限に掛かるのは売買代金が下の銘柄（上位から詰める）
        self.assertEqual(sel[sel["sector33"] == "電気機器"]["ticker"].tolist(),
                         ["1000.T", "1001.T", "1002.T"])

    def test_unknown_sector_is_not_capped(self):
        out = self._passed(12)
        sel = select_candidates(out, {})   # 業種が取れない
        self.assertEqual(len(sel), 12)

    def test_only_passing_rows_are_selected(self):
        df = make_rows(20)
        out, _meta = apply_e1(df, log=QUIET)
        sel = select_candidates(out, {})
        self.assertEqual(len(sel), 18)
        self.assertNotIn("1019.T", sel["ticker"].tolist())

    def test_empty(self):
        self.assertEqual(len(select_candidates(make_rows(0), {})), 0)


class EvaluateUniverseTest(unittest.TestCase):
    def test_skips_short_history_and_missing(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        close = pd.Series(np.linspace(300.0, 700.0, 300), index=idx)
        df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                           "Close": close, "Volume": 1e6}, index=idx)
        ohlcv = {"1111.T": df, "2222.T": df.iloc[:30]}
        out = evaluate_universe(ohlcv, ["1111.T", "2222.T", "3333.T"], close, 3, log=QUIET)
        self.assertEqual(out["ticker"].tolist(), ["1111.T"])
        self.assertEqual(list(out.columns), SCREEN_COLS)

    def test_empty_universe_returns_columns(self):
        out = evaluate_universe({}, [], pd.Series(dtype=float), 3, log=QUIET)
        self.assertEqual(list(out.columns), SCREEN_COLS)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()


class MonitoringTest(unittest.TestCase):
    """運用の監視に使う集計（docs/SCREENER.md §3.6）。条件も閾値も変えない、観測だけ。"""

    def test_fail_counts_counts_non_true(self):
        df = make_rows(10)
        df.loc[0:2, "C1"] = False
        df.loc[0:0, "D2"] = pd.NA        # 欠損も不成立として数える
        counts = fail_counts(df)
        self.assertEqual(counts["C1"], 3)
        self.assertEqual(counts["D2"], 1)
        self.assertEqual(counts["A1"], 0)
        self.assertNotIn("E1", counts)   # E1 は母集団の外では未判定なので含めない

    def test_fail_counts_on_empty(self):
        counts = fail_counts(make_rows(0))
        self.assertEqual(set(counts), set(SELF_CONTAINED_IDS))
        self.assertEqual(sum(counts.values()), 0)

    def test_landing_ma_breakdown_keys_are_stable(self):
        df = make_rows(6)
        df.loc[0:1, "landing_ma"] = "SMA5"
        df.loc[4:5, "landing_ma"] = "SMA200"
        counts = landing_ma_breakdown(df)
        self.assertEqual(list(counts), ["SMA5", "SMA25", "SMA75", "SMA200"])
        self.assertEqual(counts, {"SMA5": 2, "SMA25": 2, "SMA75": 0, "SMA200": 2})

    def test_landing_ma_breakdown_ignores_missing(self):
        df = make_rows(4)
        df["landing_ma"] = ""
        self.assertEqual(sum(landing_ma_breakdown(df).values()), 0)
        self.assertEqual(sum(landing_ma_breakdown(make_rows(0)).values()), 0)

    def test_format_counts(self):
        counts = {"A1": 1, "B2": 5, "C3": 3}
        self.assertEqual(format_counts(counts), "B2:5 C3:3 A1:1")
        self.assertEqual(format_counts(counts, top=2), "B2:5 C3:3")
        self.assertEqual(format_counts(counts, sort=False), "A1:1 B2:5 C3:3")

    def test_build_and_save_summary(self):
        df = make_rows(20)
        df.loc[0:3, "C1"] = False
        evaluated, meta = apply_e1(df, log=QUIET)
        candidates = select_candidates(evaluated, {})
        summary = build_summary(evaluated, candidates, meta,
                                pd.Timestamp("2026-08-28"), pd.Timestamp("2026-08-31"))
        self.assertEqual(summary["asof"], "2026-08-28")
        self.assertEqual(summary["delivered_on"], "2026-08-31")
        self.assertEqual(summary["n_evaluated"], 20)
        self.assertEqual(summary["n_pool"], 16)
        self.assertEqual(summary["n_candidates"], len(candidates))
        self.assertFalse(summary["e1_skipped"])
        self.assertIsNotNone(summary["e1_threshold"])
        self.assertEqual(summary["fail_counts"]["C1"], 4)
        self.assertEqual(sum(summary["landing_ma_all"].values()), 20)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_summary(summary, Path(tmp), pd.Timestamp("2026-08-31"))
            self.assertEqual(path.name, "screen_summary_2026-08-31.json")
            self.assertEqual(path, summary_path(Path(tmp), pd.Timestamp("2026-08-31")))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)

    def test_summary_records_e1_skip(self):
        evaluated, meta = apply_e1(make_rows(E1_MIN_POOL - 1), log=QUIET)
        summary = build_summary(evaluated, select_candidates(evaluated, {}), meta,
                                pd.Timestamp("2026-08-28"), pd.Timestamp("2026-08-31"))
        self.assertTrue(summary["e1_skipped"])
        self.assertIsNone(summary["e1_threshold"])   # JSON に NaN を書かない

    def test_summary_on_empty_day(self):
        evaluated, meta = apply_e1(make_rows(0), log=QUIET)
        summary = build_summary(evaluated, select_candidates(evaluated, {}), meta, None,
                                pd.Timestamp("2026-08-31"))
        self.assertIsNone(summary["asof"])
        self.assertEqual(summary["n_candidates"], 0)
        self.assertTrue(summary["e1_skipped"])
