"""モメンタム・スクリーニング — configuration (env-driven).

歪み系(mispricing/)とは完全に独立。全振り運用を前提に、以下をハード制約として実装:
- 実保有は最大3銘柄(ユーザー明示指定・資金量制約)
- セクター分散必須(同一業種は1銘柄まで)
- リスク%は確信度で可変にせず固定0.5%(3銘柄×0.5%=合計1.5%で総リスク上限2.0%内に収まる設計)
- レジームフィルター(TOPIX)は例外なく厳守
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
    outdir: str = field(default_factory=lambda: os.getenv("REPORT_OUTDIR", "out_momentum"))

    # --- 流動性フィルター(歪み系と同じ基準) ---
    min_price: float = field(default_factory=lambda: _f("MIN_PRICE", 100.0))
    min_adv_jpy: float = field(default_factory=lambda: _f("MIN_ADV_JPY", 5.0e8))

    # --- ★ポジション構成のハード制約(ユーザー明示指定) ---
    max_positions: int = field(default_factory=lambda: _i("MOM_MAX_POSITIONS", 3))
    max_per_sector: int = field(default_factory=lambda: _i("MOM_MAX_PER_SECTOR", 1))
    max_watch: int = field(default_factory=lambda: _i("MOM_MAX_WATCH", 2))  # 本命3+参考2=表示5
    risk_pct_fixed: float = field(default_factory=lambda: _f("MOM_RISK_PCT_FIXED", 0.5))
    # ★指示②(既定OFF): "fixed"=ユーザー指定のハード制約どおり固定0.5%を維持。
    # "atr_scaled"に切り替えるとATR%に応じて0.5%を可変化する(要ユーザー確認の上で有効化すること)。
    sizing_mode: str = field(default_factory=lambda: os.getenv("MOM_SIZING_MODE", "fixed"))
    atr_scale_min: float = field(default_factory=lambda: _f("ATR_SCALE_MIN", 0.6))
    atr_scale_max: float = field(default_factory=lambda: _f("ATR_SCALE_MAX", 1.5))
    total_risk_cap: float = field(default_factory=lambda: _f("TOTAL_RISK_CAP", 2.0))
    account_equity: float = field(default_factory=lambda: _f("ACCOUNT_EQUITY", 0.0))
    min_exec_jpy: float = field(default_factory=lambda: _f("MIN_EXEC_JPY", 1.0e6))

    # --- STEP1: レジームフィルター(TOPIX、絶対遵守) ---
    regime_ticker_primary: str = field(default_factory=lambda: os.getenv("REGIME_TICKER", "1306.T"))
    regime_ticker_fallback: str = field(default_factory=lambda: os.getenv("REGIME_TICKER_FALLBACK", "^N225"))
    regime_sma_days: int = field(default_factory=lambda: _i("REGIME_SMA_DAYS", 150))  # 指示⑤a: 200→150(反応速度優先)
    regime_mom_days: int = field(default_factory=lambda: _i("REGIME_MOM_DAYS", 252))

    # --- STEP2: 候補プール(広く維持) ---
    pool_size: int = field(default_factory=lambda: _i("MOM_POOL_SIZE", 100))  # 60〜120目安
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 420))
    data_coverage_min: float = field(default_factory=lambda: _f("DATA_COVERAGE_MIN", 0.70))

    # --- モメンタム総合スコアの重み ---
    w_mom_12_1: float = field(default_factory=lambda: _f("W_MOM_12_1", 0.40))
    w_high52w: float = field(default_factory=lambda: _f("W_HIGH52W", 0.25))
    w_relstrength: float = field(default_factory=lambda: _f("W_RELSTRENGTH", 0.25))
    w_trend_align: float = field(default_factory=lambda: _f("W_TREND_ALIGN", 0.10))
    # ★セクター強度加点(新規・価格データのみで機械的に算出。ニュース等は使わない)
    w_sector_strength: float = field(default_factory=lambda: _f("W_SECTOR_STRENGTH", 0.15))
    sector_strength_min_members: int = field(default_factory=lambda: _i("SECTOR_STRENGTH_MIN_MEMBERS", 3))

    # --- STEP3: 3状態分類 ---
    adx_trend_th: float = field(default_factory=lambda: _f("ADX_TREND_TH", 25.0))
    adx_period: int = field(default_factory=lambda: _i("ADX_PERIOD", 14))
    # 状態A(すでに流入): 押し目ゾーン
    pullback_sma_fast: int = field(default_factory=lambda: _i("PULLBACK_SMA_FAST", 10))
    pullback_sma_slow: int = field(default_factory=lambda: _i("PULLBACK_SMA_SLOW", 20))
    pullback_tolerance_pct: float = field(default_factory=lambda: _f("PULLBACK_TOLERANCE_PCT", 2.5))
    # ★調査(2026-07-11)反映: 「±2.5%」対称帯を非対称帯に置換(20日線基準・下側を広く)
    pullback_upper_pct: float = field(default_factory=lambda: _f("PULLBACK_UPPER_PCT", 2.5))   # MAの上+2.5%まで
    pullback_lower_pct: float = field(default_factory=lambda: _f("PULLBACK_LOWER_PCT", 5.0))   # MAの下-5.0%まで
    # 深さ上限(追加): ATR倍数上限 + 50日線からの下限%。Alajbeg et al.(2017)「MAから大きく下は最低リターン」に対応
    pullback_depth_atr_mult: float = field(default_factory=lambda: _f("PULLBACK_DEPTH_ATR_MULT", 3.0))
    pullback_ma50_floor_pct: float = field(default_factory=lambda: _f("PULLBACK_MA50_FLOOR_PCT", 10.0))
    swing_high_lookback_days: int = field(default_factory=lambda: _i("SWING_HIGH_LOOKBACK_DAYS", 60))
    # 継続期間上限(追加): 短期リバーサルの窓(概ね1ヶ月)を超えたらトレンド劣化とみなす
    pullback_max_duration_days: int = field(default_factory=lambda: _i("PULLBACK_MAX_DURATION_DAYS", 20))
    # 健全性フィルター(追加): 52週高値近接度。George & Hwang(2004)。モメンタムクラッシュのloser側を除外
    health_high52w_min: float = field(default_factory=lambda: _f("HEALTH_HIGH52W_MIN", 0.75))
    # ★押し目の反発確認(新規): ゾーン内にいるだけでなく実際に反発し始めていることを要求する
    bounce_lookback_days: int = field(default_factory=lambda: _i("BOUNCE_LOOKBACK_DAYS", 2))
    # ★調査(2026-07-11)反映: CLV≥0.5(伝統的な-1〜+1のClose Location Value)は
    # close_position(0〜1スケール)に換算すると0.75に相当(CLV=2×close_position-1)。0.5→0.75へ厳格化。
    bounce_min_close_position: float = field(default_factory=lambda: _f("BOUNCE_MIN_CLOSE_POSITION", 0.75))
    # 状態B(初動・VCPブレイク)
    vcp_lookback: int = field(default_factory=lambda: _i("VCP_LOOKBACK", 20))
    vcp_contraction_ratio: float = field(default_factory=lambda: _f("VCP_CONTRACTION_RATIO", 0.75))
    vcp_breakout_vol_mult: float = field(default_factory=lambda: _f("VCP_BREAKOUT_VOL_MULT", 1.5))
    donchian_days: int = field(default_factory=lambda: _i("DONCHIAN_DAYS", 20))
    # 状態C(流出)
    breakdown_sma: int = field(default_factory=lambda: _i("BREAKDOWN_SMA", 50))

    # --- TOB/コーポレートアクション疑いの検出(価格ヒューリスティック) ---
    tob_lookback_days: int = field(default_factory=lambda: _i("TOB_LOOKBACK_DAYS", 250))  # 90→250(古い案件対策)
    tob_jump_threshold: float = field(default_factory=lambda: _f("TOB_JUMP_THRESHOLD", 0.15))
    tob_vol_collapse_ratio: float = field(default_factory=lambda: _f("TOB_VOL_COLLAPSE_RATIO", 0.35))
    # 単日ジャンプ非依存の持続的ボラ圧縮検出(MBO等の緩やかな値動き対策・新規)
    tob_sustained_recent_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_RECENT_DAYS", 20))
    tob_sustained_baseline_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_BASELINE_DAYS", 60))
    tob_sustained_vol_ratio: float = field(default_factory=lambda: _f("TOB_SUSTAINED_VOL_RATIO", 0.25))
    # 直近の絶対的な値動きの薄さ(相対比較なしのシンプルな直接判定・新規)
    tob_flat_recent_days: int = field(default_factory=lambda: _i("TOB_FLAT_RECENT_DAYS", 15))
    tob_flat_vol_threshold: float = field(default_factory=lambda: _f("TOB_FLAT_VOL_THRESHOLD", 0.0015))

    # --- 保有銘柄監視(指示⑥⑦) ---
    score_drop_sd_threshold: float = field(default_factory=lambda: _f("SCORE_DROP_SD_THRESHOLD", 1.0))
    position_tob_jump_threshold: float = field(default_factory=lambda: _f("POSITION_TOB_JUMP_THRESHOLD", 0.15))
    position_tob_confirm_days: int = field(default_factory=lambda: _i("POSITION_TOB_CONFIRM_DAYS", 3))
    position_tob_baseline_days: int = field(default_factory=lambda: _i("POSITION_TOB_BASELINE_DAYS", 20))

    # --- entry_score自動転記の信頼性ガード(指示⑭) ---
    entry_score_max_gap_bdays: int = field(default_factory=lambda: _i("ENTRY_SCORE_MAX_GAP_BDAYS", 2))

    # --- レジーム非対称ヒステリシス(指示⑧) ---
    regime_confirm_days: int = field(default_factory=lambda: _i("REGIME_CONFIRM_DAYS", 2))
    regime_history_path: str = field(default_factory=lambda: os.getenv(
        "REGIME_HISTORY_PATH", "momentum_regime_history.csv"))

    # --- z-score業種偏り診断(指示⑨・診断のみ、スコア式は変更しない) ---
    sector_diag_top_n: int = field(default_factory=lambda: _i("SECTOR_DIAG_TOP_N", 5))

    # --- 出口設計:シャンデリア・トレーリング + 初期ストップ ---
    atr_period: int = field(default_factory=lambda: _i("ATR_PERIOD", 22))
    chandelier_mult: float = field(default_factory=lambda: _f("CHANDELIER_MULT", 3.0))
    initial_stop_atr_mult: float = field(default_factory=lambda: _f("INITIAL_STOP_ATR_MULT", 2.0))
    max_risk_width_pct: float = field(default_factory=lambda: _f("MAX_RISK_WIDTH_PCT", 10.0))

    # --- 出力 ---
    reject_backfill_bdays: int = field(default_factory=lambda: _i("REJECT_BACKFILL_BDAYS", 5))
    font_path: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))


def load_config() -> Config:
    return Config()
