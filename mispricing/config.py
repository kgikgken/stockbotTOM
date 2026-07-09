"""v5.0 歪み×資金循環 screener — configuration (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


def _b(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    # --- runtime ---
    dryrun: bool = field(default_factory=lambda: _b("SCREEN_DRYRUN", False))
    outdir: str = field(default_factory=lambda: os.getenv("REPORT_OUTDIR", "out"))

    # --- universe / liquidity prefilter (ADV5億円・ロット/ADV比は自動充足) ---
    min_price: float = field(default_factory=lambda: _f("MIN_PRICE", 100.0))
    min_adv_jpy: float = field(default_factory=lambda: _f("MIN_ADV_JPY", 5.0e8))
    lot_jpy: float = field(default_factory=lambda: _f("LOT_JPY", 1.5e6))
    max_lot_adv_ratio: float = field(default_factory=lambda: _f("MAX_LOT_ADV_RATIO", 0.01))

    # --- Gate1: normalized primary triggers (補助・絶対値/RSI) ---
    z_dev_th: float = field(default_factory=lambda: _f("Z_DEV_TH", 2.0))
    pctl_th: float = field(default_factory=lambda: _f("PCTL_TH", 5.0))
    rsi_long: float = field(default_factory=lambda: _f("RSI_LONG", 30.0))
    rsi_short: float = field(default_factory=lambda: _f("RSI_SHORT", 70.0))
    rsi_ext_long: float = field(default_factory=lambda: _f("RSI_EXT_LONG", 20.0))
    rsi_ext_short: float = field(default_factory=lambda: _f("RSI_EXT_SHORT", 80.0))

    # --- Gate1: bucket fallback thresholds (絶対乖離%・業種系列が作れない場合) ---
    bucket_dev: dict = field(default_factory=lambda: {
        "Prime": _f("BUCKET_DEV_PRIME", 12.0),
        "Standard": _f("BUCKET_DEV_STANDARD", 15.0),
        "Growth": _f("BUCKET_DEV_GROWTH", 18.0),
    })
    bucket_rsi_long: dict = field(default_factory=lambda: {
        "Prime": _f("BUCKET_RSI_L_PRIME", 27.0),
        "Standard": _f("BUCKET_RSI_L_STANDARD", 25.0),
        "Growth": _f("BUCKET_RSI_L_GROWTH", 22.0),
    })
    bucket_rsi_short: dict = field(default_factory=lambda: {
        "Prime": _f("BUCKET_RSI_S_PRIME", 73.0),
        "Standard": _f("BUCKET_RSI_S_STANDARD", 75.0),
        "Growth": _f("BUCKET_RSI_S_GROWTH", 78.0),
    })

    # --- Engine A: 業種内リバーサル(v5.0) ---
    rel_z_th: float = field(default_factory=lambda: _f("REL_Z_TH", 2.0))
    rel_pctl_th: float = field(default_factory=lambda: _f("REL_PCTL_TH", 5.0))
    dip_min_days: int = field(default_factory=lambda: _i("DIP_MIN_DAYS", 2))
    dip_max_days: int = field(default_factory=lambda: _i("DIP_MAX_DAYS", 5))
    event_overshoot_mult: float = field(default_factory=lambda: _f("EVENT_OVERSHOOT_MULT", 1.25))
    bucket_reldev: dict = field(default_factory=lambda: {
        "Prime": _f("BUCKET_RELDEV_PRIME", 8.0),
        "Standard": _f("BUCKET_RELDEV_STANDARD", 10.0),
        "Growth": _f("BUCKET_RELDEV_GROWTH", 12.0),
    })
    min_sector_members: int = field(default_factory=lambda: _i("MIN_SECTOR_MEMBERS", 3))

    # --- Engine S(逆流戻り売り) / Engine B(決算月) ---
    rebound_min_days: int = field(default_factory=lambda: _i("REBOUND_MIN_DAYS", 2))
    rebound_max_days: int = field(default_factory=lambda: _i("REBOUND_MAX_DAYS", 3))
    rebound_min_atr: float = field(default_factory=lambda: _f("REBOUND_MIN_ATR", 1.0))
    engine_b_months: tuple = field(default_factory=lambda: tuple(
        int(x) for x in os.getenv("ENGINE_B_MONTHS", "2,5,8,11").split(",")))
    pead_lookback: int = field(default_factory=lambda: _i("PEAD_LOOKBACK", 15))
    pead_max_drift_pct: float = field(default_factory=lambda: _f("PEAD_MAX_DRIFT_PCT", 10.0))

    # --- 資金循環マップ / レジーム ---
    flow_window: int = field(default_factory=lambda: _i("FLOW_WINDOW", 5))
    flow_window_long: int = field(default_factory=lambda: _i("FLOW_WINDOW_LONG", 10))
    regime_up_share: float = field(default_factory=lambda: _f("REGIME_UP_SHARE", 0.72))
    regime_dn_share: float = field(default_factory=lambda: _f("REGIME_DN_SHARE", 0.28))
    regime_rotation_spread: float = field(default_factory=lambda: _f("REGIME_ROTATION_SPREAD", 4.0))

    # --- コスト / ネットR / 最小実行サイズ ---
    est_cost_pct: float = field(default_factory=lambda: _f("EST_COST_PCT", 0.15))
    net_r_floor: float = field(default_factory=lambda: _f("NET_R_FLOOR", 1.6))
    min_exec_jpy: float = field(default_factory=lambda: _f("MIN_EXEC_JPY", 1.0e6))
    max_watch: int = field(default_factory=lambda: _i("MAX_WATCH", 3))

    # --- catalyst signature (非ファンダ性の痕跡判定) ---
    event_ret_sigma: float = field(default_factory=lambda: _f("EVENT_RET_SIGMA", 2.5))
    event_vol_z: float = field(default_factory=lambda: _f("EVENT_VOL_Z", 2.5))
    event_lookback: int = field(default_factory=lambda: _i("EVENT_LOOKBACK", 5))

    # --- R床 / 到達確率 / 出口 ---
    stop_atr_mult: float = field(default_factory=lambda: _f("STOP_ATR_MULT", 1.5))
    stop_atr_mult_hivol: float = field(default_factory=lambda: _f("STOP_ATR_MULT_HIVOL", 2.0))
    stop_buffer_pct: float = field(default_factory=lambda: _f("STOP_BUFFER_PCT", 0.3))
    max_risk_width_pct: float = field(default_factory=lambda: _f("MAX_RISK_WIDTH_PCT", 8.0))
    hold_days: int = field(default_factory=lambda: _i("HOLD_DAYS", 5))
    expiry_days: int = field(default_factory=lambda: _i("CANDIDATE_EXPIRY_DAYS", 3))
    trail_days: int = field(default_factory=lambda: _i("TRAIL_DAYS", 3))

    # --- 確信度→リスク% / 総リスク上限 ---
    risk_pct_high: float = field(default_factory=lambda: _f("RISK_PCT_HIGH", 1.0))
    risk_pct_mid: float = field(default_factory=lambda: _f("RISK_PCT_MID", 0.5))
    total_risk_cap: float = field(default_factory=lambda: _f("TOTAL_RISK_CAP", 2.0))
    total_risk_cap_half: float = field(default_factory=lambda: _f("TOTAL_RISK_CAP_HALF", 1.0))
    account_equity: float = field(default_factory=lambda: _f("ACCOUNT_EQUITY", 0.0))

    # --- 地合い ---
    vi_half_lot: float = field(default_factory=lambda: _f("VI_HALF_LOT", 30.0))
    vi_warn: float = field(default_factory=lambda: _f("VI_WARN", 28.0))
    vi_severe: float = field(default_factory=lambda: _f("VI_SEVERE", 35.0))
    nikkei_vi_manual: float = field(default_factory=lambda: _f("NIKKEI_VI_VALUE", 0.0))
    hivol_atr_pct: float = field(default_factory=lambda: _f("HIVOL_ATR_PCT", 4.0))

    # --- data quality ---
    data_coverage_min: float = field(default_factory=lambda: _f("DATA_COVERAGE_MIN", 0.70))
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 420))

    # --- output ---
    max_candidates: int = field(default_factory=lambda: _i("MAX_CANDIDATES", 5))
    max_runners_up: int = field(default_factory=lambda: _i("MAX_RUNNERS_UP", 5))
    reject_backfill_bdays: int = field(default_factory=lambda: _i("REJECT_BACKFILL_BDAYS", 5))
    font_path: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))


def load_config() -> Config:
    return Config()
