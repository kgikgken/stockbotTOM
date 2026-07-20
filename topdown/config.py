"""新スクリーニング(topdown) — configuration (env-driven).

2026-07-13の一新版チャットプロンプト(地合い→セクター→カタリスト→最大5銘柄)をbotに移植したもの。
旧3システム(momentum/swing2w/mispricing)からの完全移行(旧コードは温存・実行対象外)。

★チャット版との差分(正直な翻訳・重要):
- カタリスト一次情報(TDnet)・出典URL・ニュースはbotでは取得不可。単日ギャップ+出来高急増を
  「カタリストの価格的痕跡」として代理し、真因は要TDnet確認と必ず明記する。
- 確信度はチャット版の「一次情報の裏付けがない銘柄に高を付けない」規律に忠実に従い、
  一次情報を構造的に確認できないbotでは全候補の上限を「中」とする。
- デイ(当日手仕舞い)は大引けデータの日次botでは実行不可能なタグのため出さない。
  保有期間タグは「短期スイング(数日〜1週間)」「スイング(1〜2週間)」の2種のみ。
- 需給(信用買い残)はデータが無いため省略(レポートに省略項目として明記)。
- 日経VIは取得不安定のためN225実現ボラ20d(年率換算)で代理(旧歪み系と同じ方式)。
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
    outdir: str = field(default_factory=lambda: os.getenv("TOPDOWN_OUTDIR", "out_topdown"))

    # --- 流動性フィルター(従来と同じ基準) ---
    min_price: float = field(default_factory=lambda: _f("MIN_PRICE", 100.0))
    # ★価格上限(2026-07-19): 1単元=100株。株価が高いと1単元すら建玉に収まらないため上限を置く。
    # 20,000円 = 1単元200万円。多くの銘柄が1単元のみとなり、その場合は半分利確を使わず
    # トレーリング+時間ストップ一本で運用する(screen/position_check側で自動判定)。
    max_price: float = field(default_factory=lambda: _f("MAX_PRICE", 20000.0))
    min_adv_jpy: float = field(default_factory=lambda: _f("MIN_ADV_JPY", 5.0e8))
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 420))
    data_coverage_min: float = field(default_factory=lambda: _f("DATA_COVERAGE_MIN", 0.70))

    # --- 本命/次点の枠 ---
    max_candidates: int = field(default_factory=lambda: _i("TOPDOWN_MAX_CANDIDATES", 5))
    max_per_sector: int = field(default_factory=lambda: _i("TOPDOWN_MAX_PER_SECTOR", 2))  # セクター整合が設計思想のため2まで許容
    max_watch: int = field(default_factory=lambda: _i("TOPDOWN_MAX_WATCH", 3))
    risk_pct_fixed: float = field(default_factory=lambda: _f("TOPDOWN_RISK_PCT_FIXED", 0.5))
    account_equity: float = field(default_factory=lambda: _f("ACCOUNT_EQUITY", 0.0))
    min_exec_jpy: float = field(default_factory=lambda: _f("MIN_EXEC_JPY", 1.0e6))

    # --- STEP1: 地合い(米国・為替・VIX・日経。ルーブリック加減点、全て仮置き) ---
    sentiment_spx_th: float = field(default_factory=lambda: _f("SENT_SPX_TH", 0.3))    # S&P500 1日騰落±%
    sentiment_sox_th: float = field(default_factory=lambda: _f("SENT_SOX_TH", 1.0))    # SOX 1日騰落±%
    sentiment_vix_high: float = field(default_factory=lambda: _f("SENT_VIX_HIGH", 25.0))
    sentiment_n225_5d_th: float = field(default_factory=lambda: _f("SENT_N225_5D_TH", 3.0))
    vi_high_threshold: float = field(default_factory=lambda: _f("VI_HIGH_THRESHOLD", 30.0))  # VI代理>30で高ボラ環境
    # 高ボラ環境での半導体・値がさ大型の扱い(改訂版ルール): SOX前夜反発なら高ボラタグ付きで採用可
    sox_rebound_th: float = field(default_factory=lambda: _f("SOX_REBOUND_TH", 1.0))
    semis_tickers: tuple = ("8035.T", "6857.T", "9984.T", "6146.T", "6920.T")  # 東エレク/アドテスト/SBG/ディスコ/レーザーテック

    # --- STEP2: セクター(構成銘柄等ウェイト代理) ---
    sector_lookback_days: int = field(default_factory=lambda: _i("SECTOR_LOOKBACK_DAYS", 5))
    sector_min_members: int = field(default_factory=lambda: _i("SECTOR_MIN_MEMBERS", 5))
    sector_top_n: int = field(default_factory=lambda: _i("SECTOR_TOP_N", 3))
    sector_bottom_n: int = field(default_factory=lambda: _i("SECTOR_BOTTOM_N", 2))

    # --- STEP3: トリガー(カタリスト代理・テクニカル。閾値は旧システムの検証済み初期値を引き継ぎ) ---
    gap_threshold: float = field(default_factory=lambda: _f("GAP_THRESHOLD", 0.04))       # 単日+4%
    gap_vol_mult: float = field(default_factory=lambda: _f("GAP_VOL_MULT", 2.0))
    gap_max_days_since: int = field(default_factory=lambda: _i("GAP_MAX_DAYS_SINCE", 1))  # ★0〜1営業日(PEADは直後最強)
    breakout_lookback: int = field(default_factory=lambda: _i("BREAKOUT_LOOKBACK", 20))   # 節目=20日高値(仮置き)
    breakout_vol_mult: float = field(default_factory=lambda: _f("BREAKOUT_VOL_MULT", 1.5))
    # S高・急騰(監視格下げ+寄り天警告)
    spike_1d_threshold: float = field(default_factory=lambda: _f("SPIKE_1D_THRESHOLD", 0.15))   # 前日比+15%
    spike_3d_threshold: float = field(default_factory=lambda: _f("SPIKE_3D_THRESHOLD", 0.25))   # 3日累積+25%
    # 押し目5ゲート(旧momentum凍結版の値をそのまま自立化・2026-07-17)
    pullback_upper_pct: float = field(default_factory=lambda: _f("PULLBACK_UPPER_PCT", 2.5))    # 20日線の上+2.5%
    pullback_lower_pct: float = field(default_factory=lambda: _f("PULLBACK_LOWER_PCT", 5.0))    # 20日線の下-5.0%
    pullback_depth_atr_mult: float = field(default_factory=lambda: _f("PULLBACK_DEPTH_ATR_MULT", 3.0))  # ATR22×3以内
    pullback_max_duration_days: int = field(default_factory=lambda: _i("PULLBACK_MAX_DURATION_DAYS", 35))
    swing_high_lookback_days: int = field(default_factory=lambda: _i("SWING_HIGH_LOOKBACK_DAYS", 60))
    bounce_lookback_days: int = field(default_factory=lambda: _i("BOUNCE_LOOKBACK_DAYS", 2))
    bounce_min_close_position: float = field(default_factory=lambda: _f("BOUNCE_MIN_CLOSE_POSITION", 0.75))  # CLV≥0.5相当

    # --- 入口ゾーン(v2.0・2026-07-19) ---
    atr_period: int = field(default_factory=lambda: _i("ATR_PERIOD", 14))
    # 材料反応: ギャップ足の高値〜値幅の38.2%押しまで(Alajbeg et al.2017/Bulkowski「浅い押し>深い押し」)
    zone_fib_retrace: float = field(default_factory=lambda: _f("ZONE_FIB_RETRACE", 0.382))
    # 高値ブレイク: ブレイク水準〜1ATR下まで
    zone_break_atr_mult: float = field(default_factory=lambda: _f("ZONE_BREAK_ATR_MULT", 1.0))
    # 構造ストップの緩衝(ヒゲ刈られ回避・仮置き)
    stop_buffer_atr_mult: float = field(default_factory=lambda: _f("STOP_BUFFER_ATR_MULT", 0.2))
    # リスク幅の上限。★合否判定ではなくゾーン上端の切り落としに使う(STOP + これ)
    max_risk_atr_mult: float = field(default_factory=lambda: _f("MAX_RISK_ATR_MULT", 1.5))
    # ★リスク幅の下限(2026-07-19追加): ゾーン下端が構造ストップに近すぎると、
    # 損切り幅が0.2ATR程度になりノイズで即刻刈られる非現実的な設計になる。
    # 下端を STOP + これ まで引き上げ、「底値ちょうどを拾いにいかない」ことを構造化する。
    min_risk_atr_mult: float = field(default_factory=lambda: _f("MIN_RISK_ATR_MULT", 0.5))
    # ゾーン未到達での失効(営業日)
    zone_expire_days: int = field(default_factory=lambda: _i("ZONE_EXPIRE_DAYS", 3))

    # --- 出口(v2.0: 固定2R利確は撤廃。2Rは参照点) ---
    reference_r: float = field(default_factory=lambda: _f("REFERENCE_R", 2.0))       # 表示上の参照点のみ
    partial_tp_r: float = field(default_factory=lambda: _f("PARTIAL_TP_R", 1.0))     # +1Rで半分利確(2単元以上時)
    time_stop_gap: int = field(default_factory=lambda: _i("TIME_STOP_GAP", 20))      # 材料反応(日本PEAD: Okada&Saeki 2014)
    time_stop_gap_max: int = field(default_factory=lambda: _i("TIME_STOP_GAP_MAX", 60))
    time_stop_break: int = field(default_factory=lambda: _i("TIME_STOP_BREAK", 10))  # 高値ブレイク
    time_stop_pull: int = field(default_factory=lambda: _i("TIME_STOP_PULL", 10))    # 押し目
    # トレーリング(残玉用・日次OHLCVで実装可能な方式)
    trail_chandelier_days: int = field(default_factory=lambda: _i("TRAIL_CHANDELIER_DAYS", 22))
    trail_chandelier_atr: float = field(default_factory=lambda: _f("TRAIL_CHANDELIER_ATR", 3.0))
    trail_close_atr_mult: float = field(default_factory=lambda: _f("TRAIL_CLOSE_ATR_MULT", 2.0))
    trail_ma_days: int = field(default_factory=lambda: _i("TRAIL_MA_DAYS", 10))

    # --- 相関集中の抑制(2026-07-19) ---
    max_gap_candidates: int = field(default_factory=lambda: _i("MAX_GAP_CANDIDATES", 3))
    max_same_gapday: int = field(default_factory=lambda: _i("MAX_SAME_GAPDAY", 2))

    # --- 決算またぎ推定(警告のみ・除外しない) ---
    earnings_est_enabled: bool = field(default_factory=lambda: _b("EARNINGS_EST_ENABLED", True))
    earnings_cycle_days: int = field(default_factory=lambda: _i("EARNINGS_CYCLE_DAYS", 60))
    earnings_spike_vol_mult: float = field(default_factory=lambda: _f("EARNINGS_SPIKE_VOL_MULT", 2.5))

    # --- TOB/コーポレートアクション疑い(momentum系の3経路検出を共有) ---
    tob_lookback_days: int = field(default_factory=lambda: _i("TOB_LOOKBACK_DAYS", 250))
    tob_jump_threshold: float = field(default_factory=lambda: _f("TOB_JUMP_THRESHOLD", 0.15))
    tob_vol_collapse_ratio: float = field(default_factory=lambda: _f("TOB_VOL_COLLAPSE_RATIO", 0.35))
    tob_sustained_recent_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_RECENT_DAYS", 20))
    tob_sustained_baseline_days: int = field(default_factory=lambda: _i("TOB_SUSTAINED_BASELINE_DAYS", 60))
    tob_sustained_vol_ratio: float = field(default_factory=lambda: _f("TOB_SUSTAINED_VOL_RATIO", 0.25))
    tob_flat_recent_days: int = field(default_factory=lambda: _i("TOB_FLAT_RECENT_DAYS", 15))
    tob_flat_vol_threshold: float = field(default_factory=lambda: _f("TOB_FLAT_VOL_THRESHOLD", 0.0015))

    # --- イベントカード ---
    events_path: str = field(default_factory=lambda: os.getenv("EVENTS_PATH", "events.csv"))
    events_horizon_days: int = field(default_factory=lambda: _i("EVENTS_HORIZON_DAYS", 7))

    # --- 出力 ---
    font_path: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: os.getenv(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))


def load_config() -> Config:
    return Config()
