"""再計算一致テストハーネス（DESIGN.md §11 / TASKS.md T-205）。

「T で切った系列から計算した値」＝「全期間で計算して T 行を取り出した値」が一致する
ことを、機械的に検査する共通ハーネス。DESIGN.md §11 冒頭のとおり、後者は「一括計算
して行を取り出す」ことではなく「各 T についてそのときまでのデータでローリング計算
する」ことを指す（T-202 の質問ログ回答で明確化済み）。

対象は features/dimensions.py に登録された全特徴量（FEATURE_IDS）＋押し目構造/状態
（h0/l0/lp/state）。特徴量は dimensions.FEATURE_IDS から動的に列挙するので、新しい
特徴量を FEATURE_METADATA に追加すれば、このテストを1行も変更せず次回実行から自動的
に検査対象へ入る（受け入れ条件）。

合成データは data/synthetic.py の make_synthetic / make_synthetic_index を再利用する
（DRYRUN 経路で実績のある生成器。上昇＋押し目・横ばい・下落・分割・ギャップの5種類を
銘柄ごとに決定的な乱数で作り分ける）。10銘柄 × 5 T点で検査する。
"""
import unittest

import numpy as np
import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.synthetic import make_synthetic, make_synthetic_index
from stockbot.features import dimensions, indicators, pullback, swings

K = 3
N_TICKERS = 10
N_BARS = 400
T_POINTS_PER_TICKER = 5
PULLBACK_STATE_FIELDS = ("state", "h0", "l0", "lp")


def run_pipeline(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, dividends: pd.Series, idx_close: pd.Series,
                 t_pos: int, k: int = K):
    """指標 → スイング → 押し目 → 特徴量、の一連のパイプラインを T 時点まで計算する。

    T-205 の「任意の特徴量関数」を横断する共通ハーネスの中核。個々のモジュール
    （indicators/swings/pullback/dimensions）は既にそれぞれの再計算一致テストを
    持つが、ここではパイプライン全体を通した一致を dimensions の全登録特徴量に
    ついて機械的に検査する。
    """
    sma5 = indicators.sma(close, 5)
    sma200 = indicators.sma(close, 200)
    atr14 = indicators.atr_wilder(high, low, close, 14)
    raw = swings.detect_raw_swings(high, low, k)
    alternated = swings.alternate_swings(raw)
    pb = pullback.pullback_state(high, low, close, sma5, sma200, atr14, alternated, t_pos, k)
    features, extra = dimensions.compute_dimensions(
        open_, high, low, close, volume, dividends, alternated, pb, t_pos, k,
        idx_close=idx_close)
    return pb, features, extra


def assert_value_equal(test: unittest.TestCase, cut_value, full_value, msg: str) -> None:
    """特徴量1件ぶんの値を比較する（bool/NaN/浮動小数を使い分ける）。

    bool(float('nan')) は True になるため、bool 判定より先に NaN 判定をする
    （full=True/cut=NaN のような食い違いを bool(NaN)=True で素通りさせない）。
    """
    full_na, cut_na = pd.isna(full_value), pd.isna(cut_value)
    if full_na or cut_na:
        test.assertEqual(cut_na, full_na, msg=f"{msg} (片方だけ NaN)")
        return
    if isinstance(full_value, (bool, np.bool_)) or isinstance(cut_value, (bool, np.bool_)):
        test.assertEqual(bool(cut_value), bool(full_value), msg=msg)
    else:
        test.assertAlmostEqual(float(cut_value), float(full_value), places=9, msg=msg)


class LookaheadConsistencyHarnessTest(unittest.TestCase):
    """合成10銘柄 × T5点で、登録済み全特徴量＋押し目構造/状態の再計算一致を検査する。"""

    @classmethod
    def setUpClass(cls):
        cls.tickers = [f"{1301 + i * 37:04d}.T" for i in range(N_TICKERS)]
        end = pd.Timestamp("2026-08-21")
        cls.ohlcv = make_synthetic(cls.tickers, n_bars=N_BARS, seed=0, end=end)
        cls.idx_close = make_synthetic_index(n_bars=N_BARS, seed=0, end=end)["Close"]
        # SMA200 等が定義される範囲から均等に5点選ぶ。末尾ちょうどだと切り詰めが no-op に
        # なり検査が無意味になるため、系列末尾より手前に上限を取る
        cls.t_points = sorted({int(x) for x in np.linspace(220, N_BARS - 20, T_POINTS_PER_TICKER)})
        for t in cls.t_points:
            assert t < N_BARS - 1, f"t_points は系列末尾より手前でなければならない: t={t}"

    def test_all_registered_dimensions_are_recalc_consistent(self):
        checked_ids: set[str] = set()
        states_seen: set[str] = set()
        n_checks = 0

        for ticker in self.tickers:
            df = self.ohlcv[ticker]
            open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
            volume, dividends = df["Volume"], df["Dividends"]

            for t in self.t_points:
                full_pb, full_feats, full_extra = run_pipeline(
                    open_, high, low, close, volume, dividends, self.idx_close, t)
                cut_pb, cut_feats, cut_extra = run_pipeline(
                    open_.iloc[:t + 1], high.iloc[:t + 1], low.iloc[:t + 1],
                    close.iloc[:t + 1], volume.iloc[:t + 1], dividends.iloc[:t + 1],
                    self.idx_close.iloc[:t + 1], t)
                states_seen.add(full_pb["state"])

                for field in PULLBACK_STATE_FIELDS:
                    self.assertEqual(cut_pb[field], full_pb[field],
                                     msg=f"{ticker} t={t} pullback.{field}")

                # 特徴量IDは dimensions.FEATURE_IDS から動的に列挙する（メタデータ由来）。
                # 新しい特徴量が追加されれば、この for ループが自動的に拾う。
                for fid in dimensions.FEATURE_IDS:
                    full_v = full_feats.loc[full_feats["id"] == fid, "value"].iloc[0]
                    cut_v = cut_feats.loc[cut_feats["id"] == fid, "value"].iloc[0]
                    assert_value_equal(self, cut_v, full_v, msg=f"{ticker} t={t} {fid}")
                    checked_ids.add(fid)
                    n_checks += 1

                self.assertEqual(cut_extra, full_extra, msg=f"{ticker} t={t} extra")

        # 受け入れ条件: dimensions.FEATURE_IDS の全件が実際に検査されたこと
        self.assertEqual(checked_ids, set(dimensions.FEATURE_IDS))
        self.assertEqual(n_checks, len(self.tickers) * len(self.t_points) * len(dimensions.FEATURE_IDS))
        # 合成データが押し目の5状態＋no_structureを一通り作っていることの確認（偏りの検知）
        self.assertGreaterEqual(len(states_seen), 4, msg=f"states_seen={states_seen}")

    def test_harness_detects_a_deliberately_broken_recalc(self):
        """ハーネス自身が壊れていないことの確認: assert_value_equal が実際に不一致を検出するか。"""
        ticker = self.tickers[0]
        df = self.ohlcv[ticker]
        close = df["Close"]
        t = self.t_points[0]

        # 「全期間の最終値」を返す未来参照ダミー特徴量。Tで切ると最終値が変わるはず
        full_leak = float(close.iloc[-1])
        cut_leak = float(close.iloc[: t + 1].iloc[-1])
        self.assertNotEqual(t, len(close) - 1)
        self.assertNotEqual(full_leak, cut_leak,
                            "このテスト自体が意味を持つには全期間値とT時点値が異なる必要がある")

        # (a) 数値の不一致
        with self.assertRaises(AssertionError):
            assert_value_equal(self, cut_leak, full_leak, msg="case a")
        # (b) full=数値 / cut=NaN
        with self.assertRaises(AssertionError):
            assert_value_equal(self, np.nan, full_leak, msg="case b")
        # (c) full=True / cut=NaN（bool(nan)==True で素通りしないことの確認）
        with self.assertRaises(AssertionError):
            assert_value_equal(self, np.nan, True, msg="case c")


if __name__ == "__main__":
    unittest.main()
