"""2週間スイング(swing2w) — configuration (env-driven).

歪み系(mispricing/)・モメンタム系(momentum/)いずれとも完全に独立した第3のエンジン。
調査レポート(2026-07-11)の中核提案を実装する:

- 2週間という保有期間は「短期反転」と「中期モメンタム」の転換点に位置し、
  どちらが効くかは回転率(turnover、ここではADVを代理指標として使用)で決まる
  (Dai, Medhat, Novy-Marx & Rizova 2023; Medhat & Schmeling 2022)。
- そのため、母集団を回転率で二分し、
  ★エンジンR(低〜中回転率 × 業種内相対で売られ過ぎ)= 反転の残りを取る
  ★エンジンM(高回転率 × 好材料由来の値動き)= モメンタムの入口だけを取る
  の2エンジンに振り分ける。
- 出口はシャンデリア・トレーリング(momentum系)ではなく、固定利確+時間ストップの
  ハイブリッドを主軸とする(平均回帰的な値幅には固定目標が有効という実証知見に基づく)。

ハード制約(ユーザー明示指定):
- 実保有はモメンタム系とは別枠で最大3銘柄
- 同一業種は1銘柄まで(このエンジン内で独立して判定。モメンタム系の保有とは合算しない)
- リスク%は固定0.5%
"""

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
    dryrun: bool = field(default_factory=lambda: _b("SCREEN_DRYRUN", False))
    outdir: str = field(default_factory=lambda: os.getenv("SWING2W_OUTDIR", "out_swing2w"))

    # --- 流動性フィルター(他系と同じ基準) ---
    min_price: float = field(default_factory=lambda: _f("MIN_PRICE", 100.0))
    min_adv_jpy: float = field(default_factory=lambda: _f("MIN_ADV_JPY", 5.0e8))
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 420))
    data_coverage_min: float = field(default_factory=lambda: _f("DATA_COVERAGE_MIN", 0.70))

    # --- ★ポジション構成のハード制約(モメンタム系とは別枠) ---
    max_positions: int = field(default_factory=lambda: _i("SWING2W_MAX_POSITIONS", 3))
    max_per_sector: int = field(default_factory=lambda: _i("SWING2W_MAX_PER_SECTOR", 1))
    max_watch: int = field(default_factory=lambda: _i("SWING2W_MAX_WATCH", 2))
    risk_pct_fixed: float = field(default_factory=lambda: _f("SWING2W_RISK_PCT_FIXED", 0.5))
    total_risk_cap: float = field(default_factory=lambda: _f("SWING2W_TOTAL_RISK_CAP", 1.5))
    account_equity: float = field(default_factory=lambda: _f("ACCOUNT_EQUITY", 0.0))
    min_exec_jpy: float = field(default_factory=lambda: _f("MIN_EXEC_JPY", 1.0e6))

    # --- 母集団の回転率二分(この設計の核心) ---
    # ADVを回転率の代理指標として使用(真の売買高/浮動株数は取得できないため)。
    # 母集団を回転率でパーセンタイル分割し、下位を低回転率(エンジンR)、
    # 上位を高回転率(エンジンM)の対象母集団とする。
    turnover_low_pct: float = field(default_factory=lambda: _f("TURNOVER_LOW_PCT", 0.40))   # 下位40%
    turnover_high_pct: float = field(default_factory=lambda: _f("TURNOVER_HIGH_PCT", 0.60))  # 上位40%(60%点以上)

    # --- エンジンR(反転・低〜中回転率) ---
    rel_lookback_days: int = field(default_factory=lambda: _i("R_REL_LOOKBACK_DAYS", 5))
    rel_oversold_z: float = field(default_factory=lambda: _f("R_REL_OVERSOLD_Z", -1.2))  # 業種内相対z値の下側閾値(安定性重視で-1.0→-1.2)
    rel_min_sector_members: int = field(default_factory=lambda: _i("R_REL_MIN_SECTOR_MEMBERS", 5))  # 3→5(統計の安定性優先)
    rsi_period: int = field(default_factory=lambda: _i("RSI_PERIOD", 14))
    rsi_secondary_th: float = field(default_factory=lambda: _f("RSI_SECONDARY_TH", 40.0))  # 補助確認(主軸ではない)
    r_min_dip_days: int = field(default_factory=lambda: _i("R_MIN_DIP_DAYS", 2))
    r_max_dip_days: int = field(default_factory=lambda: _i("R_MAX_DIP_DAYS", 5))
    # ★調査(2026-07-11)反映: z値基準は維持しつつ、深さレンジ(浅すぎ/深すぎを除外)を追加。
    # 「50日線の-5%〜-15%」または「ATR2〜5倍の下落」のいずれかを満たすことを要求する。
    r_depth_ma50_min_pct: float = field(default_factory=lambda: _f("R_DEPTH_MA50_MIN_PCT", 5.0))
    r_depth_ma50_max_pct: float = field(default_factory=lambda: _f("R_DEPTH_MA50_MAX_PCT", 15.0))
    r_depth_atr_min: float = field(default_factory=lambda: _f("R_DEPTH_ATR_MIN", 2.0))
    r_depth_atr_max: float = field(default_factory=lambda: _f("R_DEPTH_ATR_MAX", 5.0))
    swing_high_lookback_days: int = field(default_factory=lambda: _i("SWING_HIGH_LOOKBACK_DAYS", 60))
    # ★押し目の反発確認(新規): ゾーン内にいるだけでなく実際に反発し始めていることを要求する(momentumと共通仕様)
    bounce_lookback_days: int = field(default_factory=lambda: _i("BOUNCE_LOOKBACK_DAYS", 2))
    bounce_min_close_position: float = field(default_factory=lambda: _f("BOUNCE_MIN_CLOSE_POSITION", 0.75))

    # --- エンジンM(モメンタム入口・高回転率) ---
    gap_threshold: float = field(default_factory=lambda: _f("M_GAP_THRESHOLD", 0.04))  # 単日+4%以上
    gap_vol_mult: float = field(default_factory=lambda: _f("M_GAP_VOL_MULT", 2.0))
    breakout_days: int = field(default_factory=lambda: _i("M_BREAKOUT_DAYS", 252))  # 52週高値
    breakout_vol_mult: float = field(default_factory=lambda: _f("M_BREAKOUT_VOL_MULT", 1.5))
    m_max_days_since_trigger: int = field(default_factory=lambda: _i("M_MAX_DAYS_SINCE_TRIGGER", 3))

    # --- 出口設計(固定利確+時間ストップのハイブリッド・シャンデリアは使わない) ---
    atr_period: int = field(default_factory=lambda: _i("ATR_PERIOD", 14))
    initial_stop_atr_mult: float = field(default_factory=lambda: _f("INITIAL_STOP_ATR_MULT", 1.75))
    profit_target_r: float = field(default_factory=lambda: _f("PROFIT_TARGET_R", 2.5))
    breakeven_trigger_r: float = field(default_factory=lambda: _f("BREAKEVEN_TRIGGER_R", 1.0))
    time_stop_days: int = field(default_factory=lambda: _i("TIME_STOP_DAYS", 10))  # 営業日
    max_risk_width_pct: float = field(default_factory=lambda: _f("MAX_RISK_WIDTH_PCT", 8.0))

    # --- TOB/コーポレートアクション疑い(momentum系と同じ考え方を流用) ---
    tob_lookback_days: int = field(default_factory=lambda: _i("TOB_LOOKBACK_DAYS", 250))
    tob_jump_threshold: float = field(default_factory=lambda: _f("TOB_JUMP_THRESHOLD", 0.15))
    tob_vol_collapse_ratio: float = field(default_factory=lambda: _f("TOB_VOL_COLLAPSE_RATIO", 0.35))
    tob_sustained_recent_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_RECENT_DAYS", 20))
    tob_sustained_baseline_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_BASELINE_DAYS", 60))
    tob_sustained_vol_ratio: float = field(default_factory=lambda: _f("TOB_SUSTAINED_VOL_RATIO", 0.25))
    tob_flat_recent_days: int = field(default_factory=lambda: _i("TOB_FLAT_RECENT_DAYS", 15))
    tob_flat_vol_threshold: float = field(default_factory=lambda: _f("TOB_FLAT_VOL_THRESHOLD", 0.0015))

    # --- 出力 ---
    font_path: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))


def load_config() -> Config:
    return Config()
