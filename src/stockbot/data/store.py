"""ローカル保存とスナップショット（SPEC §4 緩和策 3）。

- store/ohlcv.csv.gz      … 縦持ち (ticker, date, open, high, low, close, volume, dividends, splits)
                            pyarrow があれば ohlcv.parquet を使う
- daily/YYYY-MM-DD.csv.gz … その日に「新規に観測した」足。日付別の小さなファイルなので
                            git にコミットしても肥大しない。後日の前向き検証と時点復元に使う
- store/revisions.csv.gz  … 既存の足が後から書き換わった記録（yfinance の遡及修正の証跡）
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
        if len(old) == 0:
            return new.sort_values(["ticker", "date"]).reset_index(drop=True), new.copy(), \
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

    def append_revisions(self, revisions: pd.DataFrame, observed_on: pd.Timestamp) -> None:
        if revisions is None or len(revisions) == 0:
            return
        rev = revisions.copy()
        rev.insert(0, "observed_on", pd.Timestamp(observed_on).strftime("%Y-%m-%d"))
        if self.revisions_path.exists():
            prev = pd.read_csv(self.revisions_path)
            rev = pd.concat([prev, rev], ignore_index=True)
        rev.to_csv(self.revisions_path, index=False, compression="gzip")
