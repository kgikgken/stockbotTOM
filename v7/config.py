"""stockbotTOM v7.1 設定 — 次元抽出型(Dimension Coverage Model)。

仕様書の原則:
  - すべてのパラメータは設定ファイルに外出しし、ハードコードしない(§3-⑤)
  - 次元ウェイトは等分。不等ウェイトは自由パラメータを生むため採用しない(§4.1)
  - 数値はすべて「検証前の出発点」であり根拠のある値ではない(第12部)

証拠階層(§0.3):
  A=日本株の査読研究あり / B=国際的査読研究あり / C=論争中 / D=実証空白 / E=否定的
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _f(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def _i(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v not in (None, "") else default
    except Exception:
        return default


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v in (None, "") else str(v).strip().lower() in {"1", "true", "yes", "on"}


def _s(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v in (None, "") else v


@dataclass
class ConfigV7:
    # ---- 実行 ----
    dryrun: bool = field(default_factory=lambda: _b("SCREEN_DRYRUN", False))
    outdir: str = field(default_factory=lambda: _s("OUTDIR", "out_v7"))
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 500))

    # ================= 第6部 時点整合 =================
    # 全次元を t−n に揃える(§6.2 規則1)。fractalの確定ラグに合わせる。
    # 「基準時点の異なる情報を単一のDCSに合成すると先読みバイアスが発生する」
    align_lag_days: int = field(default_factory=lambda: _i("V7_ALIGN_LAG", 3))

    # ---- 検出器: fractal(ZigZag不採用・§6.3) ----
    # 確定タイミングが全銘柄で一定になるためfractalを採用。
    fractal_n: int = field(default_factory=lambda: _i("V7_FRACTAL_N", 3))          # 日足
    fractal_n_weekly: int = field(default_factory=lambda: _i("V7_FRACTAL_N_W", 2))  # 週足

    # ================= L0 レジーム判定 =================
    # L0-A【主】指数トレンド(階層A: Hanauer 2014)
    l0_weekly_ma: int = field(default_factory=lambda: _i("V7_L0_WEEKLY_MA", 30))
    l0_slope_weeks: int = field(default_factory=lambda: _i("V7_L0_SLOPE_WEEKS", 5))
    l0_hysteresis_band: float = field(default_factory=lambda: _f("V7_L0_HYST_BAND", 0.03))
    l0_confirm_days: int = field(default_factory=lambda: _i("V7_L0_CONFIRM_DAYS", 5))
    # L0-B【補助】ディストリビューションデー(階層E)
    # 仕様書§5.1.2 規則3: 日本株で有意性が確認されるまで寄与度ゼロでよい
    l0_dd_enabled: bool = field(default_factory=lambda: _b("V7_L0_DD_ENABLED", False))
    l0_dd_window: int = field(default_factory=lambda: _i("V7_L0_DD_WINDOW", 25))
    l0_dd_drop_pct: float = field(default_factory=lambda: _f("V7_L0_DD_DROP", 0.002))
    l0_dd_cluster: int = field(default_factory=lambda: _i("V7_L0_DD_CLUSTER", 4))

    # ================= L1 ユニバース =================
    # Amihud ILLIQ(階層A)を主指標とする(§5.2.2)
    l1_illiq_percentile: float = field(default_factory=lambda: _f("V7_L1_ILLIQ_PCT", 70.0))
    l1_illiq_window: int = field(default_factory=lambda: _i("V7_L1_ILLIQ_WINDOW", 60))
    l1_min_adv_jpy: float = field(default_factory=lambda: _f("V7_L1_MIN_ADV", 5.0e7))  # 慣行値5,000万
    l1_min_price: float = field(default_factory=lambda: _f("V7_L1_MIN_PRICE", 100.0))
    l1_min_listed_days: int = field(default_factory=lambda: _i("V7_L1_MIN_LISTED", 210))  # 30週分

    # ================= 次元共通(§3.0・§4) =================
    theta: float = field(default_factory=lambda: _f("V7_THETA", 0.60))
    coverage_main: float = field(default_factory=lambda: _f("V7_COVERAGE_MAIN", 0.80))
    coverage_watch: float = field(default_factory=lambda: _f("V7_COVERAGE_WATCH", 0.60))
    # 次元の有効/無効。検証で増分がなければ「削除」する(§1.5・減重しない)
    dim1_enabled: bool = field(default_factory=lambda: _b("V7_DIM1", True))
    dim2_enabled: bool = field(default_factory=lambda: _b("V7_DIM2", True))
    dim3_enabled: bool = field(default_factory=lambda: _b("V7_DIM3", True))
    dim4_enabled: bool = field(default_factory=lambda: _b("V7_DIM4", True))
    dim5_enabled: bool = field(default_factory=lambda: _b("V7_DIM5", True))

    # ---- ① トレンド方向 ----
    d1_ma_short: int = field(default_factory=lambda: _i("V7_D1_MA_S", 50))
    d1_ma_mid: int = field(default_factory=lambda: _i("V7_D1_MA_M", 150))
    d1_ma_long: int = field(default_factory=lambda: _i("V7_D1_MA_L", 200))
    d1_slope_window: int = field(default_factory=lambda: _i("V7_D1_SLOPE_WIN", 20))
    # 1-e/1-f は階層E。仕様書§3-①注記により、含む版/除く版の比較検証を前提に切替可能とする
    d1_use_52w_components: bool = field(default_factory=lambda: _b("V7_D1_USE_52W", True))
    d1_low52_gain: float = field(default_factory=lambda: _f("V7_D1_LOW52_GAIN", 0.30))
    d1_high52_gap: float = field(default_factory=lambda: _f("V7_D1_HIGH52_GAP", 0.25))
    d1_weekly_ma_a: int = field(default_factory=lambda: _i("V7_D1_WMA_A", 30))
    d1_weekly_ma_b: int = field(default_factory=lambda: _i("V7_D1_WMA_B", 60))
    # 小次郎ステージのスコア(1-h・階層D)
    d1_stage_scores: dict = field(default_factory=lambda: {
        1: 1.0, 6: 0.7, 5: 0.4, 2: 0.3, 3: 0.1, 4: 0.0})
    d1_stage_short: int = field(default_factory=lambda: _i("V7_D1_ST_S", 5))
    d1_stage_mid: int = field(default_factory=lambda: _i("V7_D1_ST_M", 20))
    d1_stage_long: int = field(default_factory=lambda: _i("V7_D1_ST_L", 40))

    # ---- ② 相対力 ----
    # 直近1ヶ月除外は階層A。「変更しない」と仕様書第12部に明記
    d2_skip_days: int = field(default_factory=lambda: _i("V7_D2_SKIP", 21))
    d2_lookback_days: int = field(default_factory=lambda: _i("V7_D2_LOOKBACK", 252))
    d2_floor_percentile: float = field(default_factory=lambda: _f("V7_D2_FLOOR_PCT", 20.0))
    d2_rs_slope_window: int = field(default_factory=lambda: _i("V7_D2_RS_SLOPE", 60))
    # リバーサル調整版/非調整版の両方を算出(Iwanaga 2024の再現テスト・§3-②)
    d2_reversal_adjust: bool = field(default_factory=lambda: _b("V7_D2_REV_ADJ", False))

    # ---- ③ 需給・出来高 ----
    d3_vol_short: int = field(default_factory=lambda: _i("V7_D3_VOL_S", 5))
    d3_vol_long: int = field(default_factory=lambda: _i("V7_D3_VOL_L", 50))
    d3_updown_window: int = field(default_factory=lambda: _i("V7_D3_UPDOWN_WIN", 50))

    # ---- ④ 時間・日柄 ----
    # IBD基準を出発点とする(§3-④・日本株の実証は存在しない)
    d4_base_min_weeks: int = field(default_factory=lambda: _i("V7_D4_BASE_MIN_W", 5))
    d4_base_ideal_weeks: int = field(default_factory=lambda: _i("V7_D4_BASE_IDEAL_W", 8))
    d4_base_max_weeks: int = field(default_factory=lambda: _i("V7_D4_BASE_MAX_W", 30))
    d4_stage_ideal_days: int = field(default_factory=lambda: _i("V7_D4_STAGE_IDEAL", 20))
    d4_stage_max_days: int = field(default_factory=lambda: _i("V7_D4_STAGE_MAX", 60))
    d4_base_threshold: float = field(default_factory=lambda: _f("V7_D4_BASE_TH", 0.03))

    # ---- ⑤ ボラティリティ収縮 ----
    d5_atr_period: int = field(default_factory=lambda: _i("V7_D5_ATR", 14))
    d5_atr_trend_window: int = field(default_factory=lambda: _i("V7_D5_ATR_TREND", 60))
    d5_bb_period: int = field(default_factory=lambda: _i("V7_D5_BB", 20))
    d5_bb_compare: int = field(default_factory=lambda: _i("V7_D5_BB_CMP", 120))
    # 原典に数値根拠なし(§3-⑤)。すべて検証対象
    d5_contract_min: int = field(default_factory=lambda: _i("V7_D5_CON_MIN", 2))
    d5_contract_max: int = field(default_factory=lambda: _i("V7_D5_CON_MAX", 6))
    d5_shrink_ratio: float = field(default_factory=lambda: _f("V7_D5_SHRINK", 0.6))
    d5_final_tight_atr: float = field(default_factory=lambda: _f("V7_D5_FINAL_TIGHT", 2.0))
    d5_w_monotonic: float = field(default_factory=lambda: _f("V7_D5_W_MONO", 1/3))
    d5_w_tightness: float = field(default_factory=lambda: _f("V7_D5_W_TIGHT", 1/3))
    d5_w_count: float = field(default_factory=lambda: _f("V7_D5_W_COUNT", 1/3))

    # ================= 出力 =================
    max_output: int = field(default_factory=lambda: _i("V7_MAX_OUTPUT", 20))
    max_watch_output: int = field(default_factory=lambda: _i("V7_MAX_WATCH", 20))

    # ---- フォント(PNG) ----
    font_path: str = field(default_factory=lambda: _s(
        "FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font_path_bold: str = field(default_factory=lambda: _s(
        "FONT_PATH_BOLD", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))

    # ================= 制約の明示(yfinance運用) =================
    # 仕様書が要求するがyfinanceでは満たせない項目。出力に恒久表示する(§10.2.3)
    data_limitations: List[str] = field(default_factory=lambda: [
        "生存バイアス未処理: 上場廃止銘柄を取得できないため、検証結果は実力より良く出る",
        "ポイントインタイム・ユニバース不可: 過去時点の上場銘柄一覧を復元できない",
        "L0-C(海外投資家フロー)・L0-D(業種別空売り比率)は未実装: データ取得不可",
        "L4(バリュエーション除外)は未実装: v5.0が存在しない",
        "L5(修飾子: 空売り残高・逆日歩)は未実装: データ取得不可",
        "配当調整なし: 高配当銘柄の②が系統的に過小評価される",
    ])


def load_config_v7() -> ConfigV7:
    return ConfigV7()
