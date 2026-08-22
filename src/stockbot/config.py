"""設定。すべて環境変数から読み、ハードコードしない（SPEC §1 実装）。

数値はいずれも「検証前の出発点」。検証L1の結果で更新する。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _f(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


JPX_LISTED_URL_DEFAULT = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
)


@dataclass(frozen=True)
class Settings:
    # ---- パス ----
    data_dir: Path
    universe_seed_csv: Path          # 旧リポジトリの universe_jpx.csv（JPX 取得失敗時の種）

    # ---- 取得 ----
    history_days: int                # ライブ取得の履歴日数（yfinance の period に使う）
    fetch_deadline_sec: int          # 取得に使ってよい上限秒数
    fetch_scope: str                 # auto | all | universe
    dryrun: bool                     # SCREEN_DRYRUN=1 で合成データ
    market_close_hhmm: str           # 東証の引け。これ以前は当日足を未確定として捨てる
    jpx_listed_url: str

    # ---- ユニバース（SPEC §1）----
    min_adv_jpy: float               # 20日平均売買代金の下限（円）
    min_price: float                 # 株価下限（円）
    min_history_bars: int            # 採点に必要な最小履歴本数
    adv_window: int                  # 売買代金の平均日数
    max_staleness_days: int          # 最終足がこれより古い銘柄はユニバースから外す

    # ---- 改訂検出 ----
    rev_close_tol: float             # 終値の相対差がこれを超えたら「改訂」
    rev_volume_tol: float            # 出来高の相対差がこれを超えたら「改訂」

    @classmethod
    def from_env(cls) -> "Settings":
        dryrun = _b("SCREEN_DRYRUN", False)
        # DRYRUN は既定で本番と別ディレクトリに書く。同じ data/ を共有すると、
        # 合成データが git コミットや Actions cache 経由で本番データに混入する
        # （2026-08-22 実運用で発生・確認済み）。DATA_DIR を明示指定すれば無効。
        default_data_dir = "data-dryrun" if dryrun else "data"
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", default_data_dir)),
            universe_seed_csv=Path(os.getenv("UNIVERSE_SEED_CSV", "universe_jpx.csv")),
            history_days=_i("HISTORY_DAYS", 400),
            fetch_deadline_sec=_i("FETCH_DEADLINE_SEC", 5400),
            fetch_scope=os.getenv("FETCH_SCOPE", "auto").strip().lower(),
            dryrun=dryrun,
            market_close_hhmm=os.getenv("MARKET_CLOSE_HHMM", "15:30"),
            jpx_listed_url=os.getenv("JPX_LISTED_URL", JPX_LISTED_URL_DEFAULT),
            min_adv_jpy=_f("MIN_ADV_JPY", 2e8),
            min_price=_f("MIN_PRICE", 200.0),
            min_history_bars=_i("MIN_HISTORY_BARS", 250),
            adv_window=_i("ADV_WINDOW", 20),
            max_staleness_days=_i("MAX_STALENESS_DAYS", 7),
            rev_close_tol=_f("REV_CLOSE_TOL", 0.005),
            rev_volume_tol=_f("REV_VOLUME_TOL", 0.05),
        )

    # ---- 派生パス ----
    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def universe_dir(self) -> Path:
        return self.data_dir / "universe"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    def ensure_dirs(self) -> None:
        for p in (self.store_dir, self.daily_dir, self.universe_dir, self.reference_dir):
            p.mkdir(parents=True, exist_ok=True)
