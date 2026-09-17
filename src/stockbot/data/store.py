"""ローカル保存とスナップショット（SPEC §4 緩和策 3）。

- store/ohlcv.csv.gz      … 縦持ち (ticker, date, open, high, low, close, volume, dividends, splits)
                            pyarrow があれば ohlcv.parquet を使う
- daily/YYYY-MM-DD.csv.gz … その日に「新規に観測した」足。日付別の小さなファイルなので
                            git にコミットしても肥大しない。後日の前向き検証と時点復元に使う
- store/revisions.csv.gz  … 既存の足が後から書き換わった記録（yfinance の遡及修正の証跡）
- store/seam_issues.csv   … 継ぎ目の検査に引っかかった銘柄（新規取得行を store に入れずに止めた記録）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

IDX_TICKER = "__IDX__"  # 指数（TOPIX/日経225）を store に保存するときの ticker 値（T-102）
LONG_COLS = ["ticker", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]
_WIDE_TO_LONG = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                 "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits"}
_LONG_TO_WIDE = {v: k for k, v in _WIDE_TO_LONG.items()}


# 継ぎ目の検査（2026-09-17・1909.T の破損を受けて追加）。
#
# **取得ウィンドウ全体が一様に再スケールされると `check_splits` は素通りする** ——
# あちらは取得したフレームの中の隣接バー比を見るので、全部が同じ倍率で狂っていると
# 段差が出ない。段差は「新しく取った行」と「取得ウィンドウより古い既存行」の**継ぎ目**
# にあり、そこは今まで誰も検査していなかった。
#
# 1909.T（2026-09-13）は終値が**一定倍率 4,399,472 倍**（時価総額らしき値）になり、
# volume が全行 0 になった。倍率の上限を 100 にしてあるのは、日本株の分割・併合は
# 大きくても 1:10 程度で、**100 倍なら分割では説明がつかない**ため（1909.T は
# これの 4 万倍以上）。分割を弾くための値ではなく、**分割では説明できない壊れ方**を
# 捕まえるための値である。
SEAM_RATIO_MAX = 100.0
# 出来高がゼロになった行がこれ以上続いたら異常とみなす（売買停止は数日で戻る）
SEAM_ZERO_VOLUME_MIN_ROWS = 20
SEAM_COLS = ["ticker", "seam_date", "prev_date", "new_close", "prev_close",
             "close_ratio", "n_new_rows", "kind"]


def seam_issues(old: pd.DataFrame, new: pd.DataFrame,
                ratio_max: float = SEAM_RATIO_MAX,
                zero_volume_min_rows: int = SEAM_ZERO_VOLUME_MIN_ROWS) -> pd.DataFrame:
    """新規取得行と既存行の**継ぎ目**を検査する（2026-09-17 追加）。

    銘柄ごとに、新しく取った行の**最初の日**の直前にある既存行と突き合わせる。

    - `close_ratio`（新しい最初の終値 ÷ 直前の既存終値）が `ratio_max` 倍を超えるか
      `1/ratio_max` 倍を下回る → **異常**
    - 新規行が `zero_volume_min_rows` 行以上あって**全行の出来高が 0**、かつ直前の
      既存行の出来高が 0 より大きい → **異常**

    **継ぎ目が無い銘柄は検査しない** —— 新規上場（既存行が無い）や、
    `upsert_replace` で既存行を先に消した全履歴再取得（＝修復の経路）では
    直前の既存行が存在しないので、ここで止まらない。

    戻り値は `SEAM_COLS` の表（0 行なら異常なし）。**判定はここでは行わない** ——
    呼び出し側が「その銘柄の新規行を入れない」を決める。
    """
    empty = pd.DataFrame(columns=SEAM_COLS)
    if old is None or new is None or len(old) == 0 or len(new) == 0:
        return empty
    rows = []
    old_g = {t: g for t, g in old.groupby("ticker", sort=False)}
    for ticker, g_new in new.groupby("ticker", sort=False):
        g_old = old_g.get(ticker)
        if g_old is None or len(g_old) == 0:
            continue
        first_new = g_new["date"].min()
        before = g_old[g_old["date"] < first_new]
        if len(before) == 0:
            continue                      # 継ぎ目が無い（新規上場・置換のあと）
        prev = before.loc[before["date"].idxmax()]
        head = g_new.loc[g_new["date"].idxmin()]
        prev_close, new_close = float(prev["close"]), float(head["close"])
        kind = ""
        ratio = float("nan")
        if np.isfinite(prev_close) and np.isfinite(new_close) and prev_close > 0 and new_close > 0:
            ratio = new_close / prev_close
            if ratio > ratio_max or ratio < 1.0 / ratio_max:
                kind = "seam_close_ratio"
        if not kind and len(g_new) >= zero_volume_min_rows:
            v_new = pd.to_numeric(g_new["volume"], errors="coerce").fillna(0)
            if (v_new == 0).all() and float(prev.get("volume") or 0) > 0:
                kind = "seam_zero_volume"
        if not kind:
            continue
        rows.append({"ticker": ticker, "seam_date": first_new, "prev_date": prev["date"],
                     "new_close": new_close, "prev_close": prev_close,
                     "close_ratio": ratio, "n_new_rows": int(len(g_new)), "kind": kind})
    return pd.DataFrame(rows, columns=SEAM_COLS) if rows else empty


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def to_long(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for t, df in ohlcv.items():
        if df is None or len(df) == 0:
            continue
        x = df.rename(columns=_WIDE_TO_LONG).copy()
        x.index.name = "date"
        x = x.reset_index()
        x.insert(0, "ticker", t)
        frames.append(x[LONG_COLS])
    if not frames:
        return pd.DataFrame(columns=LONG_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def from_long(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or len(df) == 0:
        return out
    for t, g in df.groupby("ticker", sort=False):
        x = g.drop(columns=["ticker"]).set_index("date").sort_index()
        x.index.name = "Date"
        out[str(t)] = x.rename(columns=_LONG_TO_WIDE)
    return out


class OhlcvStore:
    def __init__(self, store_dir: Path, daily_dir: Path,
                 close_tol: float = 0.005, volume_tol: float = 0.05):
        self.store_dir = Path(store_dir)
        self.daily_dir = Path(daily_dir)
        self.close_tol = close_tol
        self.volume_tol = volume_tol
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.use_parquet = _has_pyarrow()
        self.path = self.store_dir / ("ohlcv.parquet" if self.use_parquet else "ohlcv.csv.gz")
        self.revisions_path = self.store_dir / "revisions.csv.gz"
        self.seam_path = self.store_dir / "seam_issues.csv"
        # 直近の `_merge` で**止めた**銘柄（継ぎ目の検査）。呼び出し側が読んで報告する
        self.last_seam_issues = pd.DataFrame(columns=SEAM_COLS)

    # ---------------------------------------------------------------- io
    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=LONG_COLS)
        if self.use_parquet:
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[LONG_COLS]

    def save(self, df: pd.DataFrame) -> None:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        if self.use_parquet:
            df.to_parquet(self.path, index=False)
        else:
            df.to_csv(self.path, index=False, compression="gzip")

    # ------------------------------------------------------------ upsert
    def upsert(self, new: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """new を既存に上書き統合する（マージ。new に無い既存の日は残る）。

        戻り値: (統合後, added=新規に観測した足, revisions=値が変わった足)
        """
        return self._merge(self.load(), new)

    def upsert_replace(self, new: pd.DataFrame, tickers) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """tickers に含まれる銘柄は、既存の全行を先に削除してから new を統合する
        （マージではなく置換）。

        全履歴再取得（history_full_days 等の固定長ウィンドウでの再取得）はマージだと、
        取得ウィンドウの外にある古い行が未調整のまま取り残され、ウィンドウの端に
        価格の段差が生じる（T-402、9900.Tの2016-02-09境界で実データにより確認）。
        再取得のたびにウィンドウが動くため、マージのままではウィンドウを伸ばしても
        再発し続ける。対象銘柄については置換にすることで、fetchできた範囲だけが
        storeに残る（履歴が短くなった銘柄はMIN_HISTORY_BARSの判定で自然に扱われる
        ため実害はない）。tickers に無い銘柄は通常どおりマージされる。
        """
        tickers = set(tickers)
        old = self.load()
        if len(old) and tickers:
            old = old[~old["ticker"].isin(tickers)]
        return self._merge(old, new)

    def _merge(self, old: pd.DataFrame, new: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        new = new[LONG_COLS].copy()
        new["date"] = pd.to_datetime(new["date"]).dt.normalize()
        self.last_seam_issues = pd.DataFrame(columns=SEAM_COLS)
        if len(old) == 0:
            return new.sort_values(["ticker", "date"]).reset_index(drop=True), new.copy(), \
                pd.DataFrame(columns=LONG_COLS + ["old_close", "old_volume"])

        # **継ぎ目の検査**（2026-09-17）。引っかかった銘柄の新規行は **store に入れない**
        # —— 壊れた値で既存の履歴を上書きするより、その銘柄が古いまま止まるほうがよい。
        # 記録は `seam_issues.csv` に残し、`store.last_seam_issues` で呼び出し側に返す
        bad = seam_issues(old, new)
        if len(bad):
            self.last_seam_issues = bad
            new = new[~new["ticker"].isin(set(bad["ticker"]))].copy()
            if len(new) == 0:
                return old.sort_values(["ticker", "date"]).reset_index(drop=True), \
                    pd.DataFrame(columns=LONG_COLS), \
                    pd.DataFrame(columns=LONG_COLS + ["old_close", "old_volume"])

        key = ["ticker", "date"]
        m = new.merge(old, on=key, how="left", suffixes=("", "_old"), indicator=True)
        added = m[m["_merge"] == "left_only"][LONG_COLS].copy()
        both = m[m["_merge"] == "both"]
        with np.errstate(divide="ignore", invalid="ignore"):
            c_rel = np.abs(both["close"] - both["close_old"]) / np.abs(both["close_old"])
            v_rel = np.abs(both["volume"] - both["volume_old"]) / np.where(both["volume_old"] > 0, both["volume_old"], np.nan)
        changed = both[(c_rel > self.close_tol) | (v_rel.fillna(0) > self.volume_tol)]
        revisions = changed[LONG_COLS].copy()
        revisions["old_close"] = changed["close_old"].to_numpy()
        revisions["old_volume"] = changed["volume_old"].to_numpy()

        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(key).reset_index(drop=True)
        return merged, added, revisions

    # --------------------------------------------------------- snapshots
    def write_daily_increments(self, added: pd.DataFrame, max_dates: int = 5) -> list[Path]:
        """新規観測した足を日付別ファイルに追記する（既存ファイルがあれば統合）。

        通常は最新1〜2日分だけが新規になる。初回バックフィルで大量の日付が新規になる
        場合は、直近 max_dates 日分のみファイル化する（古い分は store にだけ入る）。
        """
        written: list[Path] = []
        if added is None or len(added) == 0:
            return written
        dates = sorted(added["date"].unique())[-max_dates:]
        for d in dates:
            day = added[added["date"] == d]
            p = self.daily_dir / f"{pd.Timestamp(d).strftime('%Y-%m-%d')}.csv.gz"
            if p.exists():
                prev = pd.read_csv(p, parse_dates=["date"])
                day = pd.concat([prev, day], ignore_index=True).drop_duplicates(
                    subset=["ticker", "date"], keep="last")
            day.sort_values("ticker").to_csv(p, index=False, compression="gzip")
            written.append(p)
        return written

    def append_seam_issues(self, issues: pd.DataFrame, observed_on: pd.Timestamp) -> None:
        """継ぎ目の検査で止めた銘柄を追記する（**消さずに積む**）。"""
        if issues is None or len(issues) == 0:
            return
        out = issues.copy()
        out.insert(0, "observed_on", pd.Timestamp(observed_on).strftime("%Y-%m-%d"))
        if self.seam_path.exists():
            prev = pd.read_csv(self.seam_path)
            out = pd.concat([prev, out], ignore_index=True)
        out.to_csv(self.seam_path, index=False)

    def append_revisions(self, revisions: pd.DataFrame, observed_on: pd.Timestamp) -> None:
        if revisions is None or len(revisions) == 0:
            return
        rev = revisions.copy()
        rev.insert(0, "observed_on", pd.Timestamp(observed_on).strftime("%Y-%m-%d"))
        if self.revisions_path.exists():
            prev = pd.read_csv(self.revisions_path)
            rev = pd.concat([prev, rev], ignore_index=True)
        rev.to_csv(self.revisions_path, index=False, compression="gzip")
