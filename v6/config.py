"""stockbotTOM v6.0 設定 — 7人の手法による直列フィルター構造。

v5系との最大の違い: スコアは通過可否に一切関与しない(順序決定のみ)。
各レイヤーはAND条件で、条件不足を点数で救済しない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else str(v).strip().lower() in {"1", "true", "yes", "on"}


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class ConfigV6:
    # ---- 実行 ----
    dryrun: bool = field(default_factory=lambda: _b("SCREEN_DRYRUN", False))
    outdir: str = field(default_factory=lambda: _s("OUTDIR", "out_v6"))
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 420))

    # ---- Layer 0: 流動性ユニバース ----
    min_price: float = field(default_factory=lambda: _f("V6_MIN_PRICE", 300.0))
    max_price: float = field(default_factory=lambda: _f("V6_MAX_PRICE", 20000.0))
    min_adv_jpy: float = field(default_factory=lambda: _f("V6_MIN_ADV_JPY", 3.0e8))
    min_listed_days: int = field(default_factory=lambda: _i("V6_MIN_LISTED_DAYS", 250))

    # ---- Layer 1: 週足ステージ(ワインスタイン) ----
    weekly_ma: int = field(default_factory=lambda: _i("V6_WEEKLY_MA", 30))
    weekly_slope_weeks: int = field(default_factory=lambda: _i("V6_WEEKLY_SLOPE_WEEKS", 5))
    weekly_slope_th: float = field(default_factory=lambda: _f("V6_WEEKLY_SLOPE_TH", 0.005))
    rs_vs_index_weeks: int = field(default_factory=lambda: _i("V6_RS_VS_INDEX_WEEKS", 26))

    # ---- Layer 2: トレンドテンプレート(ミネルヴィニ) ----
    tt_low_mult: float = field(default_factory=lambda: _f("V6_TT_LOW_MULT", 1.30))
    tt_high_mult: float = field(default_factory=lambda: _f("V6_TT_HIGH_MULT", 0.75))
    tt_ma200_lookback: int = field(default_factory=lambda: _i("V6_TT_MA200_LOOKBACK", 22))
    rs_rating_min: float = field(default_factory=lambda: _f("V6_RS_RATING_MIN", 70.0))

    # ---- Layer 3: 日足ステージ(小次郎講師) ----
    stage_short: int = field(default_factory=lambda: _i("V6_STAGE_SHORT", 5))
    stage_mid: int = field(default_factory=lambda: _i("V6_STAGE_MID", 20))
    stage_long: int = field(default_factory=lambda: _i("V6_STAGE_LONG", 40))

    # ---- Layer 4-A: VCP ----
    vcp_lookback: int = field(default_factory=lambda: _i("V6_VCP_LOOKBACK", 130))
    vcp_swing_min_bars: int = field(default_factory=lambda: _i("V6_VCP_SWING_MIN_BARS", 3))
    vcp_min_contractions: int = field(default_factory=lambda: _i("V6_VCP_MIN_CONTRACTIONS", 2))
    vcp_shrink_ratio: float = field(default_factory=lambda: _f("V6_VCP_SHRINK_RATIO", 0.85))
    # 仕様書§6-Aの定義値(0.12)。
    # ※2026-08-06に一度0.065へ下げたが、2026-08-07に仕様書の定義へ差し戻した。
    # 経緯: 最終収縮が深いVCPは、8%ストップ上限(§8-1)により棄却される。これを
    # 「VCPが点灯しない不具合」と捉えて上限を下げたが、仕様書§8-1は
    # 「構造ストップが8%より深い銘柄は、そもそもリスクリワードが成立していないため棄却する。
    # ストップを緩めて無理に取りにいかない」と明記しており、棄却されること自体が
    # 意図された動作である。検出側を絞るのは仕様外の介入だったため元に戻した。
    vcp_final_depth_max: float = field(default_factory=lambda: _f("V6_VCP_FINAL_DEPTH_MAX", 0.12))
    vcp_vol_dry_ratio: float = field(default_factory=lambda: _f("V6_VCP_VOL_DRY_RATIO", 0.80))
    vcp_base_min_days: int = field(default_factory=lambda: _i("V6_VCP_BASE_MIN_DAYS", 15))
    vcp_base_max_days: int = field(default_factory=lambda: _i("V6_VCP_BASE_MAX_DAYS", 130))

    # ---- Layer 4-B: 押し目 ----
    pull_depth_min: float = field(default_factory=lambda: _f("V6_PULL_DEPTH_MIN", 0.03))
    pull_depth_max: float = field(default_factory=lambda: _f("V6_PULL_DEPTH_MAX", 0.12))
    pull_atr_max: float = field(default_factory=lambda: _f("V6_PULL_ATR_MAX", 2.5))
    pull_days_min: int = field(default_factory=lambda: _i("V6_PULL_DAYS_MIN", 5))
    pull_days_max: int = field(default_factory=lambda: _i("V6_PULL_DAYS_MAX", 25))
    pull_ma_days: int = field(default_factory=lambda: _i("V6_PULL_MA_DAYS", 25))
    pull_ma_tolerance: float = field(default_factory=lambda: _f("V6_PULL_MA_TOLERANCE", 0.98))

    # ---- Layer 4-C: ピボットブレイク ----
    pivot_vol_mult: float = field(default_factory=lambda: _f("V6_PIVOT_VOL_MULT", 1.5))
    pivot_max_extension: float = field(default_factory=lambda: _f("V6_PIVOT_MAX_EXT", 0.03))
    pivot_close_position: float = field(default_factory=lambda: _f("V6_PIVOT_CLOSE_POS", 0.6))


    # ---- Layer 5: 執行(LoK) ----
    entry_stop_buy_mult: float = field(default_factory=lambda: _f("V6_ENTRY_STOP_BUY_MULT", 1.001))
    order_expire_days: int = field(default_factory=lambda: _i("V6_ORDER_EXPIRE_DAYS", 3))

    # ---- Layer 6: リスク管理(エルダー+ミネルヴィニ) ----
    atr_period: int = field(default_factory=lambda: _i("V6_ATR_PERIOD", 14))
    stop_atr_mult: float = field(default_factory=lambda: _f("V6_STOP_ATR_MULT", 0.5))
    hard_stop_pct: float = field(default_factory=lambda: _f("V6_HARD_STOP_PCT", 0.92))
    risk_per_trade: float = field(default_factory=lambda: _f("V6_RISK_PER_TRADE", 0.02))
    max_position_pct: float = field(default_factory=lambda: _f("V6_MAX_POSITION_PCT", 0.25))
    portfolio_heat_max: float = field(default_factory=lambda: _f("V6_PORTFOLIO_HEAT_MAX", 0.06))
    dd_reduce_th: float = field(default_factory=lambda: _f("V6_DD_REDUCE_TH", 0.05))
    dd_halt_th: float = field(default_factory=lambda: _f("V6_DD_HALT_TH", 0.10))
    equity: float = field(default_factory=lambda: _f("V6_EQUITY", 10_000_000.0))
    unit_shares: int = field(default_factory=lambda: _i("V6_UNIT_SHARES", 100))

    # ---- 決算跨ぎ ----
    earnings_min_gap_days: int = field(default_factory=lambda: _i("V6_EARNINGS_MIN_GAP", 5))

    # ---- Layer 7: 出口 ----
    exit_ema_days: int = field(default_factory=lambda: _i("V6_EXIT_EMA_DAYS", 10))
    exit_ema_consecutive: int = field(default_factory=lambda: _i("V6_EXIT_EMA_CONSEC", 2))
    exit_time_stop_days: int = field(default_factory=lambda: _i("V6_EXIT_TIME_STOP", 25))
    exit_time_stop_r: float = field(default_factory=lambda: _f("V6_EXIT_TIME_STOP_R", 0.5))
    partial_tp_r: float = field(default_factory=lambda: _f("V6_PARTIAL_TP_R", 2.0))
    partial_tp_frac: float = field(default_factory=lambda: _f("V6_PARTIAL_TP_FRAC", 1.0 / 3.0))

    # ---- 通知 ----
    max_notify: int = field(default_factory=lambda: _i("V6_MAX_NOTIFY", 5))
    max_per_sector: int = field(default_factory=lambda: _i("V6_MAX_PER_SECTOR", 2))

    # ---- フォント(PNG用) ----
    font_path: str = field(default_factory=lambda: _s(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: _s(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))


def load_config_v6() -> ConfigV6:
    return ConfigV6()
