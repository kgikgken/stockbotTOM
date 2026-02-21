from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.util import (
    download_history_bulk,
    safe_float,
    is_abnormal_stock,
    atr14,
    efficiency_ratio,
    choppiness_index,
    adx,
    clamp,
)
from utils.setup import build_setup_info, liquidity_filters
from utils.rr_ev import calc_ev, pass_thresholds
from utils.diversify import apply_sector_cap, apply_corr_filter
# NOTE: We import the module (not individual names) to avoid ImportError
# when upgrading/merging branches (e.g. rs_pct_min_by_market added later).
from utils import screen_logic as _sl
from utils.blackout import blackout_reason, load_blackouts_from_env

def _rs_pct_min_fallback(_mkt_score: int) -> int:
    """Fallback for rs_pct_min_by_market (when older screen_logic is loaded)."""
    try:
        s = int(_mkt_score)
    except Exception:
        s = 60
    if s >= 70:
        return 50
    if s >= 60:
        return 55
    if s >= 50:
        return 60
    return 70

def _rs_comp_min_fallback(_mkt_score: int) -> float:
    """Fallback for rs_comp_min_by_market (when older screen_logic is loaded)."""
    try:
        s = int(_mkt_score)
    except Exception:
        s = 60
    if s >= 70:
        return -0.5
    if s >= 60:
        return 0.0
    if s >= 50:
        return 0.8
    return 1.5

# Backward-compatible accessors (provide safe fallbacks).
no_trade_conditions = getattr(_sl, 'no_trade_conditions', lambda mkt_score, delta3, macro_warning=False: False)
max_display = getattr(_sl, 'max_display', lambda macro_warning=False: 5)
rs_pct_min_by_market = getattr(_sl, 'rs_pct_min_by_market', _rs_pct_min_fallback)
rs_comp_min_by_market = getattr(_sl, 'rs_comp_min_by_market', _rs_comp_min_fallback)
from utils.saucer import scan_saucers
from utils.state import (
    in_cooldown,
    set_cooldown_days,
    record_paper_trade,
    update_paper_trades_with_ohlc,
    kpi_distortion,
)

UNIVERSE_PATH = "universe_jpx.csv"
EARNINGS_EXCLUDE_DAYS = 3  # 暦日近似（±3日）
MAX_RISK_PCT = 8.0  # リスク幅（%）がこの値以上の候補は除外（事故率低下）

# 狙える形（1〜7営業日）の上質化：
# - 期待値(CAGR寄与度)に加えて、出来高/ボラ/ギャップ等の「ノイズ」を評価
# - 重大ノイズ（イベント起因・滑り地雷）を除外し、並び順でも品質を優先
QUALITY_WEIGHT = 0.60
NOISE_EXCLUDE_SCORE = 3  # 3以上は「地雷」寄りとして除外

# 上質化（狙える形）追加：
# - RS（市場に対する相対強度）：指数を上回る銘柄を優先
# - 週足トレンド整合：日足だけ良く見えても、上位足が崩れている銘柄を落とす
# - 実行可能性：現値からエントリー帯までの距離が遠すぎる候補を落とす（機会損失を抑える）
RS20_EXCLUDE = -6.0          # RS20 がこれ未満なら基本除外（市場に負けている）
MAX_PULLBACK_ATR = 2.5      # エントリー帯までの距離が ATR の何倍まで許容するか
WEEKLY_STRICT_SETUPS = ("A1-Strong", "A1")


# Liquidity / board-thin guardrails for "狙える形"
# NOTE: We cannot observe order book directly from daily OHLCV, so we proxy it with traded value (Close*Volume).
# The user's feedback indicates "板が薄い" even when ADV is ~5億. Therefore we introduce *tiered* liquidity
# and default to showing only the "板厚" tier when available.
#
# - adv20: mean traded value over last 20 sessions (¥)
# - mdv20: median traded value over last 20 sessions (¥) to avoid spike-driven illusions
# - dv_cv20: coefficient of variation of traded value (spike detector)
# - amihud_bps100m: Amihud illiquidity scaled to bps per 1億円 (impact proxy)
#
# Tiers:
#   grade=2 (板厚): strict thresholds
#   grade=1 (準):   relaxed thresholds (used only when no grade=2 exists)
LIQ_STRICT_ADV_MIN = 800e6           # 8.0億/日
LIQ_RELAX_ADV_MIN = 500e6            # 5.0億/日
LIQ_STRICT_MDV_FACTOR = 0.75
LIQ_RELAX_MDV_FACTOR = 0.70
LIQ_STRICT_DV_CV_MAX = 1.8
LIQ_RELAX_DV_CV_MAX = 2.0
LIQ_STRICT_AMIHUD_MAX_BPS100M = 100.0
LIQ_RELAX_AMIHUD_MAX_BPS100M = 120.0

# If any strict-liquidity candidates exist, display only those to avoid "板薄" lists.
LIQ_STRICT_ONLY_IF_EXISTS = True


def _ret_n(close: pd.Series, n: int) -> float:
    """n本騰落（%）"""
    try:
        n = int(n)
    except Exception:
        n = 20
    if close is None or len(close) < n + 1:
        return float("nan")
    base = safe_float(close.iloc[-(n + 1)], np.nan)
    last = safe_float(close.iloc[-1], np.nan)
    if not (np.isfinite(base) and np.isfinite(last) and base > 0):
        return float("nan")
    return float((last / base - 1.0) * 100.0)


def _weekly_trend_ok(df: pd.DataFrame) -> bool | None:
    """週足トレンドの整合（軽量）

    True/False/None(None=データ不足)
    - 週足終値 > 週MA10 > 週MA20
    - 週MA10 が4週前より上（緩い上向き）
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    try:
        c = df["Close"].astype(float)
        wc = c.resample("W-FRI").last().dropna()
    except Exception:
        return None
    if wc is None or wc.empty or len(wc) < 30:
        return None
    ma10 = wc.rolling(10).mean()
    ma20 = wc.rolling(20).mean()
    last = safe_float(wc.iloc[-1], np.nan)
    m10 = safe_float(ma10.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m10_4 = safe_float(ma10.shift(4).iloc[-1], np.nan)
    if not (np.isfinite(last) and np.isfinite(m10) and np.isfinite(m20) and np.isfinite(m10_4)):
        return None
    ok = bool((last > m10 > m20) and (m10 >= m10_4))
    return ok




def _env_float(name: str, default: float) -> float:
    """Read a float-like environment variable with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _trend_template_metrics(df: pd.DataFrame) -> Dict[str, float | bool]:
    """Compute a simple 'trend template' score for momentum/trend screening.

    This is intentionally lightweight (daily OHLCV only) and designed for short-term
    swing *trend-following* screening. It loosely follows well-known trend-template
    ideas (MA alignment + rising long MA + proximity to highs).

    Returns keys:
      - score: 0.0..1.0
      - ok: bool
      - dist_52w_high: % distance from 52w high (0=at high)
      - from_52w_low: % above 52w low
      - ma50: last MA50
      - ma200: last MA200
      - ma200_slope20: % change of MA200 over ~1 month
      - ma_align: 1 if MA50>MA200 else 0
    """
    out: Dict[str, float | bool] = {
        "score": float("nan"),
        "ok": False,
        "dist_52w_high": float("nan"),
        "from_52w_low": float("nan"),
        "ma50": float("nan"),
        "ma200": float("nan"),
        "ma200_slope20": float("nan"),
        "ma_align": 0.0,
    }
    try:
        c = df["Close"].astype(float).dropna()
        if len(c) < 210:
            return out

        close_last = safe_float(c.iloc[-1], np.nan)

        look = int(min(_env_float("TREND_LOOKBACK_DAYS", 252.0), float(len(c))))
        if look < 60:
            return out

        hh = safe_float(c.iloc[-look:].max(), np.nan)
        ll = safe_float(c.iloc[-look:].min(), np.nan)

        ma50 = safe_float(c.rolling(50).mean().iloc[-1], np.nan)
        ma200_s = c.rolling(200).mean()
        ma200 = safe_float(ma200_s.iloc[-1], np.nan)
        ma200_prev = safe_float(ma200_s.shift(20).iloc[-1], np.nan)
        ma200_slope20 = float("nan")
        if np.isfinite(ma200) and np.isfinite(ma200_prev) and ma200_prev > 0:
            ma200_slope20 = (ma200 / ma200_prev - 1.0) * 100.0

        dist_high = float("nan")
        from_low = float("nan")
        if np.isfinite(hh) and hh > 0 and np.isfinite(close_last):
            dist_high = (hh - close_last) / hh * 100.0
        if np.isfinite(ll) and ll > 0 and np.isfinite(close_last):
            from_low = (close_last / ll - 1.0) * 100.0

        close_gt_ma50 = bool(np.isfinite(close_last) and np.isfinite(ma50) and close_last > ma50)
        close_gt_ma200 = bool(np.isfinite(close_last) and np.isfinite(ma200) and close_last > ma200)
        ma_align = bool(np.isfinite(ma50) and np.isfinite(ma200) and ma50 > ma200)
        ma200_up = bool(np.isfinite(ma200_slope20) and ma200_slope20 >= 0.0)

        # Scoring (sum to 1.0)
        score = 0.0
        if close_gt_ma50:
            score += 0.20
        if close_gt_ma200:
            score += 0.10
        if ma_align:
            score += 0.20
        if ma200_up:
            score += 0.20

        max_dist_high = _env_float("TREND_MAX_DIST_52W_HIGH", 25.0)
        min_from_low = _env_float("TREND_MIN_FROM_52W_LOW", 30.0)

        if np.isfinite(dist_high) and dist_high <= max_dist_high:
            score += 0.20
        if np.isfinite(from_low) and from_low >= min_from_low:
            score += 0.10

        score = float(min(1.0, max(0.0, score)))
        ok = bool(score >= _env_float("TREND_TEMPLATE_MIN_SCORE", 0.70))

        out.update(
            {
                "score": score,
                "ok": ok,
                "dist_52w_high": dist_high,
                "from_52w_low": from_low,
                "ma50": ma50,
                "ma200": ma200,
                "ma200_slope20": ma200_slope20,
                "ma_align": 1.0 if ma_align else 0.0,
            }
        )
        return out
    except Exception:
        return out


def _volume_dry_ratio(df: pd.DataFrame, lookback: int = 10) -> float:
    """Down-volume / Up-volume ratio over the recent window.

    For healthy pullbacks in an uptrend, we generally prefer down-volume <= up-volume.
    Values > 1.0 can indicate distribution / supply.
    """
    try:
        v = df["Volume"].astype(float)
        c = df["Close"].astype(float)
        if len(v) < lookback + 2:
            return float("nan")
        v_win = v.iloc[-(lookback + 1):]
        c_win = c.iloc[-(lookback + 1):]
        d = c_win.diff()
        up = v_win[d > 0]
        dn = v_win[d < 0]
        if len(up) < 2 or len(dn) < 2:
            return float("nan")
        up_m = safe_float(up.mean(), np.nan)
        dn_m = safe_float(dn.mean(), np.nan)
        if not (np.isfinite(up_m) and np.isfinite(dn_m)) or up_m <= 0:
            return float("nan")
        return float(dn_m / up_m)
    except Exception:
        return float("nan")


def _gap_atr_metrics(df: pd.DataFrame, lookback: int = 60, atr_mult: float = 1.0) -> Dict[str, float]:
    """Overnight gap risk using ATR units.

    Computes frequency and max magnitude of |Open - prevClose| in ATR units
    over the recent window.
    """
    out = {"freq": float("nan"), "max": float("nan")}
    try:
        if len(df) < lookback + 30:
            return out
        o = df["Open"].astype(float)
        c = df["Close"].astype(float)
        atr_s = atr14(df).shift(1)  # use yesterday's ATR
        gap = (o - c.shift(1)).abs()
        ratio = gap / atr_s.replace(0.0, np.nan)
        win = ratio.iloc[-lookback:]
        freq = float((win > float(atr_mult)).mean())
        mx = safe_float(win.max(), np.nan)
        out["freq"] = freq
        out["max"] = mx
        return out
    except Exception:
        return out


def _bb_width_ratio(df: pd.DataFrame, lookback: int = 60) -> float:
    """Bollinger Band width contraction ratio (20d width / median(20d width, lookback)).

    < 1.0 indicates contraction (tight), > 1.0 indicates expansion (loose).
    """
    try:
        c = df["Close"].astype(float)
        if len(c) < 120:
            return float("nan")
        ma20 = c.rolling(20).mean()
        sd20 = c.rolling(20).std()
        width = (4.0 * sd20) / ma20.replace(0.0, np.nan)  # (upper-lower)/ma
        w_last = safe_float(width.iloc[-1], np.nan)
        w_med = safe_float(width.iloc[-lookback:].median(), np.nan)
        if not (np.isfinite(w_last) and np.isfinite(w_med)) or w_med <= 0:
            return float("nan")
        return float(w_last / w_med)
    except Exception:
        return float("nan")



# --- Additional helpers for higher-precision trend screening ---

def _linreg_slope_r2(series: pd.Series, lookback: int) -> tuple[float, float]:
    """Linear-regression slope & R^2 on log-price.

    Returns:
      slope_ret: approximate total return over the window (e.g. 0.05 = +5%)
      r2: trend "smoothness" (0..1)

    Why:
      - A positive slope alone is not enough; we also want a reasonably
        straight trend (higher R^2) to avoid choppy, mean-reverting names.
    """
    try:
        s = series.dropna().astype(float)
        if len(s) < lookback + 2:
            return float('nan'), float('nan')
        y = np.log(s.tail(lookback).values)
        x = np.arange(len(y), dtype=float)
        # slope in log space per bar
        b1, b0 = np.polyfit(x, y, 1)
        yhat = b1 * x + b0
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        slope_ret = float(np.expm1(b1 * (len(y) - 1)))
        return slope_ret, float(clamp(r2, 0.0, 1.0))
    except Exception:
        return float('nan'), float('nan')


def _up_down_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Up-volume / down-volume ratio (lookback bars).

    Values > 1.0 suggest accumulation (up days have more volume).
    """
    try:
        if df is None or df.empty or len(df) < lookback + 2:
            return float('nan')
        c = df['Close'].astype(float)
        v = df['Volume'].astype(float)
        ret = c.diff()
        win = slice(-lookback, None)
        up = v[win][ret[win] > 0].sum()
        dn = v[win][ret[win] < 0].sum()
        return float(up / (dn + 1e-9))
    except Exception:
        return float('nan')


def _distribution_days(df: pd.DataFrame, lookback: int = 20) -> int:
    """Count distribution days (price down with higher-than-average volume).

    This is a classic O'Neil-style concept. We use a simple variant:
      - Close down vs previous close
      - Volume > 20d average volume
    """
    try:
        if df is None or df.empty or len(df) < lookback + 2:
            return 0
        c = df['Close'].astype(float)
        v = df['Volume'].astype(float)
        prev = c.shift(1)
        vol20 = v.rolling(20).mean()
        win = slice(-lookback, None)
        cond = (c[win] < prev[win]) & (v[win] > vol20[win])
        return int(cond.sum())
    except Exception:
        return 0


def _liquidity_grade(
    adv20_yen: float,
    mdv20_yen: float,
    dv_cv20: float,
    amihud_bps100m: float,
) -> Tuple[int, float]:
    """Liquidity tiering for board-thin avoidance.

    Returns:
      (grade, adv_min_used)
        grade=2: strict (板厚)
        grade=1: relaxed (準)
        grade=0: fail
    """
    adv = float(adv20_yen) if np.isfinite(adv20_yen) else float("nan")
    mdv = float(mdv20_yen) if np.isfinite(mdv20_yen) else float("nan")
    cv = float(dv_cv20) if np.isfinite(dv_cv20) else float("nan")
    imp = float(amihud_bps100m) if np.isfinite(amihud_bps100m) else float("nan")

    def _pass(adv_min: float, mdv_factor: float, cv_max: float, imp_max: float) -> bool:
        if not (np.isfinite(adv) and adv >= float(adv_min)):
            return False
        if not (np.isfinite(mdv) and mdv >= float(adv_min) * float(mdv_factor)):
            return False
        # Spiky traded value is unreliable liquidity unless the name is very liquid.
        if np.isfinite(cv) and cv > float(cv_max) and adv < float(adv_min) * 2.0:
            return False
        if np.isfinite(imp) and imp > float(imp_max):
            return False
        return True

    if _pass(
        LIQ_STRICT_ADV_MIN,
        LIQ_STRICT_MDV_FACTOR,
        LIQ_STRICT_DV_CV_MAX,
        LIQ_STRICT_AMIHUD_MAX_BPS100M,
    ):
        return 2, float(LIQ_STRICT_ADV_MIN)
    if _pass(
        LIQ_RELAX_ADV_MIN,
        LIQ_RELAX_MDV_FACTOR,
        LIQ_RELAX_DV_CV_MAX,
        LIQ_RELAX_AMIHUD_MAX_BPS100M,
    ):
        return 1, float(LIQ_RELAX_ADV_MIN)
    return 0, float(LIQ_RELAX_ADV_MIN)

def _apply_setup_mix(cands: List[Dict], max_n: int) -> List[Dict]:
    """Enforce strategy mix per spec (when alternatives exist).

    - Pullback bucket: A1-Strong / A1 / A2
    - Breakout bucket: B
    Rule:
      - If breakout candidates exist, cap pullback to 3 and include up to 2 breakouts.
      - If no breakout candidates, allow pullbacks to fill all slots.
    """
    if max_n <= 0:
        return []
    pullbacks = [c for c in cands if c.get("setup") in ("A1-Strong", "A1", "A2")]
    breakouts = [c for c in cands if c.get("setup") == "B"]
    if not breakouts:
        return cands[:max_n]
    out: List[Dict] = []
    out.extend(pullbacks[: min(3, max_n)])
    if len(out) < max_n:
        out.extend(breakouts[: min(2, max_n - len(out))])
    # fill remaining with best remaining
    if len(out) < max_n:
        used = set([x.get("ticker") for x in out])
        for c in cands:
            if c.get("ticker") in used:
                continue
            out.append(c)
            if len(out) >= max_n:
                break
    return out


def _get_ticker_col(df: pd.DataFrame) -> str:
    if "ticker" in df.columns:
        return "ticker"
    if "code" in df.columns:
        return "code"
    return ""

def _filter_earnings(uni: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in uni.columns:
        return uni
    d = pd.to_datetime(uni["earnings_date"], errors="coerce").dt.date
    uni = uni.copy()
    keep = []
    for x in d:
        if x is None or pd.isna(x):
            keep.append(True)
            continue
        try:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return uni[keep]


def _filter_blackouts(uni: pd.DataFrame, today_date) -> pd.DataFrame:
    """Optional manual blackout filter (earnings / major event / etc.).

    Controlled by env:
      - BLACKOUT_CSV: path to CSV (default: data/blackout.csv if exists)
      - BLACKOUT_BEFORE_DAYS / BLACKOUT_AFTER_DAYS
      - BLACKOUT_EXCLUDE: 1=exclude from universe (default), 0=keep but annotate

    The CSV is offline-friendly; see utils/blackout.py.
    """

    events, bdays, adays = load_blackouts_from_env(str(today_date))
    if not events or uni is None or len(uni) == 0:
        return uni

    exclude = os.getenv("BLACKOUT_EXCLUDE", "1").strip().lower() not in ("0", "false", "no", "off")

    tcol = _get_ticker_col(uni)
    if not tcol:
        return uni

    reasons: list[str] = []
    keep: list[bool] = []
    for t in uni[tcol].tolist():
        r = blackout_reason(str(t), today_date, events, bdays, adays)
        reasons.append(str(r) if r else "")
        keep.append(False if (exclude and r) else True)

    out = uni.copy()
    out["blackout_reason"] = reasons
    if exclude:
        out = out[pd.Series(keep, index=out.index)].reset_index(drop=True)
    return out


def run_screen(
    today_str: str,
    today_date,
    mkt_score: int,
    delta3: float,
    macro_on: bool,
    state: Dict,
) -> Tuple[List[Dict], Dict, Dict[str, pd.DataFrame]]:
    """
    戻り: (final_candidates_for_line, debug_meta, ohlc_map)
    ※ 歪みEVは内部処理のみ（LINE非表示）
    """
    if not os.path.exists(UNIVERSE_PATH):
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    tcol = _get_ticker_col(uni)
    if not tcol:
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    uni = _filter_earnings(uni, today_date)
    uni = _filter_blackouts(uni, today_date)
    tickers = uni[tcol].astype(str).tolist()
    ohlc_map = download_history_bulk(tickers, period="780d", auto_adjust=True, group_size=200)

    # --- Index telemetry (for relative strength / RS)
    # yfinance is unreliable for ^TOPX, so we prefer stable Japan ETFs.
    # Default:
    #   1306.T (TOPIX ETF) -> ^N225 (Nikkei) -> ^TOPX
    rs_bench_syms = [s.strip() for s in os.getenv("RS_BENCH_TICKERS", "1306.T,^N225,^TOPX").split(",") if s.strip()]
    topx_ret20 = float("nan")
    topx_ret60 = float("nan")
    topx_ret120 = float("nan")
    bench_sym = ""
    bench_df = None
    bench_close: pd.Series | None = None

    try:
        idx_map = download_history_bulk(rs_bench_syms, period="780d", auto_adjust=True, group_size=50)
        for sym in rs_bench_syms:
            dfi = idx_map.get(sym)
            if dfi is not None and (not dfi.empty) and ("Close" in dfi.columns) and len(dfi) >= 140:
                bench_sym = sym
                bench_df = dfi
                break
        if bench_df is not None and (not bench_df.empty) and ("Close" in bench_df.columns):
            cidx = bench_df["Close"].astype(float)
            bench_close = cidx
            topx_ret20 = _ret_n(cidx, 20)
            topx_ret60 = _ret_n(cidx, 60)
            topx_ret120 = _ret_n(cidx, 120)
    except Exception:
        # If index fetch fails, RS features are simply disabled.
        topx_ret20 = float("nan")
        topx_ret60 = float("nan")
        topx_ret120 = float("nan")
        bench_sym = ""
        bench_df = None
        bench_close = None

    # paper trade update
    update_paper_trades_with_ohlc(state, "tier0_exception", ohlc_map, today_str)
    update_paper_trades_with_ohlc(state, "distortion", ohlc_map, today_str)

    # distortion KPI -> auto OFF
    kpi = kpi_distortion(state)
    if kpi["count"] >= 10:
        if (kpi["median_r"] < -0.10) or (kpi["exp_gap"] < -0.30) or (kpi["neg_streak"] >= 3):
            set_cooldown_days(state, "distortion_until", days=4)

    no_trade = no_trade_conditions(int(mkt_score), float(delta3))

    cands: List[Dict] = []
    gu_cnt = 0

    # Cross-sectional RS percentile pool (computed after scanning)
    rs_pool_syms: List[str] = []
    rs_pool_vals: List[float] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue
        df = ohlc_map.get(ticker)
        if df is None or df.empty or len(df) < 120:
            continue

        if is_abnormal_stock(df):
            continue

        ok_liq, price, adv, atrp = liquidity_filters(df)
        if not ok_liq:
            continue

        info = build_setup_info(df, macro_on=macro_on)
        if info is None:
            continue
        if info.setup == "NONE":
            continue

        if info.gu:
            gu_cnt += 1

        info.adv20 = float(adv)
        info.atrp = float(atrp)

        # Liquidity refinement (board-thin proxy)
        # Even if a stock passes the base liquidity_filters, it can still have a thin order book.
        # Use traded value (Close*Volume) statistics + Amihud impact to tier liquidity.
        dv20 = (df["Close"].astype(float) * df["Volume"].astype(float)).tail(20).dropna()
        mdv20 = safe_float(dv20.median(), np.nan)
        dv_cv20 = safe_float((dv20.std() / (dv20.mean() + 1e-9)), np.nan)

        # Price impact proxy (Amihud illiquidity)
        # approx: mean(|ret| / traded_value). We scale it into bps per 1億円 so it's interpretable.
        amihud_bps100m = float("nan")
        try:
            ret_abs20 = df["Close"].astype(float).pct_change(fill_method=None).tail(20).abs()
            ill = (ret_abs20 / (dv20 + 1e-9)).replace([np.inf, -np.inf], np.nan)
            amihud = safe_float(ill.mean(), np.nan)
            if np.isfinite(amihud):
                amihud_bps100m = float(amihud * 1e8 * 10000.0)
        except Exception:
            amihud_bps100m = float("nan")

        liq_grade, liq_adv_min = _liquidity_grade(adv, mdv20, dv_cv20, amihud_bps100m)
        if int(liq_grade) <= 0:
            continue

        # Hard quality exclusions (avoid event-driven gap mines / unstable expansion)
        vr_q = float(info.vol_ratio) if getattr(info, "vol_ratio", None) is not None else np.nan
        gf_q = float(info.gap_freq) if getattr(info, "gap_freq", None) is not None else np.nan
        ac_q = float(info.atr_contr) if getattr(info, "atr_contr", None) is not None else np.nan
        if np.isfinite(gf_q) and gf_q >= 0.30:
            continue
        if np.isfinite(vr_q) and vr_q >= 2.50:
            continue
        if np.isfinite(ac_q) and ac_q >= 1.80:
            continue

        ev = calc_ev(info, mkt_score=int(mkt_score), macro_on=macro_on)
        ok, _ = pass_thresholds(info, ev)
        if not ok:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # Precompute entry/SL risk metrics for display and sorting
        close_last = float(df["Close"].iloc[-1])
        entry_low = float(info.entry_low)
        entry_high = float(info.entry_high)
        entry_price = float(info.entry_price if info.entry_price is not None else (entry_low + entry_high) / 2.0)
        sl = float(info.sl)
        # リスク幅（%）
        # - entry band がある以上、最悪ケース（=band上限）でもルールを満たす必要がある
        # - 表示は中央(=entry_price)を維持しつつ、除外判定は band上限 を使用
        risk_pct_mid = float((entry_price - sl) / entry_price * 100.0) if entry_price > 0 else 0.0
        risk_pct_low = float((entry_low - sl) / entry_low * 100.0) if entry_low > 0 else risk_pct_mid
        risk_pct_high = float((entry_high - sl) / entry_high * 100.0) if entry_high > 0 else risk_pct_mid
        risk_pct = float(risk_pct_mid)

        # リスク幅フィルタ（表示・採用対象から除外）
        # - 8%超はギャップ/滑りで想定損失が破綻しやすいため、候補自体を落とす
        if risk_pct_high >= MAX_RISK_PCT:
            continue



        # エントリー帯までの距離（%）：実行可能性（fillability）を優先するための補助指標
        band_dist_pct = 0.0
        if close_last > entry_high and close_last > 0:
            band_dist_pct = float((close_last - entry_high) / close_last * 100.0)
        elif close_last < entry_low and close_last > 0:
            band_dist_pct = float((entry_low - close_last) / close_last * 100.0)

        # --- 実行可能性（距離を ATR で評価）
        # band_dist_pct だけだと銘柄ごとのボラ差が吸収できない。
        # 「エントリー帯まで何ATR動く必要があるか」を見て、遠すぎる候補を落とす。
        pb_atr = float("nan")
        atr_abs = float(0.0)
        if close_last > 0 and np.isfinite(atrp) and float(atrp) > 0:
            atr_abs = float(close_last * float(atrp) / 100.0)
        if atr_abs > 0 and entry_low > 0 and entry_high > 0:
            diff = 0.0
            if close_last > entry_high:
                diff = float(close_last - entry_high)
            elif close_last < entry_low:
                diff = float(entry_low - close_last)
            pb_atr = float(diff / atr_abs)
            if np.isfinite(pb_atr) and pb_atr > MAX_PULLBACK_ATR:
                continue

        # --- 週足整合（上位足が崩れているものを落とす）
        weekly_ok = _weekly_trend_ok(df)
        if weekly_ok is False and info.setup in WEEKLY_STRICT_SETUPS:
            continue

        # 品質スコア（過剰最適化を避けつつ、偽物を落とすための軽い補正）
        #  - 出来高: pullback/base では「減っている」方が綺麗（=売り圧が枯れやすい）
        #  - ボラ: 収縮している方が、ブレイク後の伸びが出やすい
        #  - ギャップ: 多い銘柄は滑り/イベント起因が多く事故りやすい
        #  - 20日騰落: 上位足の勢いがある方を優先（短期の1〜7日想定）
        quality = 0.0
        vr = float(info.vol_ratio) if getattr(info, "vol_ratio", None) is not None else np.nan
        ac = float(info.atr_contr) if getattr(info, "atr_contr", None) is not None else np.nan
        gf = float(info.gap_freq) if getattr(info, "gap_freq", None) is not None else np.nan
        r20 = float(info.ret20) if getattr(info, "ret20", None) is not None else np.nan
        rc = float(info.range_contr) if getattr(info, "range_contr", None) is not None else np.nan


        # --- Multi-angle screening metrics (trend template / volatility regime / gap risk / volume dry-up) ---
        trend_tpl = _trend_template_metrics(df)
        trend_score = safe_float(trend_tpl.get("score"), np.nan)
        if not np.isfinite(trend_score):
            continue
        dist_52w_high = safe_float(trend_tpl.get("dist_52w_high"), np.nan)
        from_52w_low = safe_float(trend_tpl.get("from_52w_low"), np.nan)

        bb_ratio = _bb_width_ratio(df, lookback=int(_env_float("BB_RATIO_LOOKBACK", 60.0)))

        vol_dry = _volume_dry_ratio(df, lookback=int(_env_float("VOL_DRY_LOOKBACK", 10.0)))

        gap_atr = _gap_atr_metrics(
            df,
            lookback=int(_env_float("GAP_ATR_LOOKBACK", 60.0)),
            atr_mult=_env_float("GAP_ATR_MULT", 1.0),
        )
        gap_atr_freq = safe_float(gap_atr.get("freq"), np.nan)
        gap_atr_max = safe_float(gap_atr.get("max"), np.nan)


        # Hard exclude: one-off extreme overnight gap (earnings/IR risk) even if freq is low.
        # This reduces 'risk mismatch' accidents (gap-through-stop) in real execution.
        if not is_pos and np.isfinite(gap_atr_max):
            gap_max_a = _env_float("GAP_ATR_MAX_LIMIT_A", 3.6)
            gap_max_b = _env_float("GAP_ATR_MAX_LIMIT_B", 4.2)
            gap_lim = gap_max_b if info.setup == "B" else gap_max_a
            # In weaker tape, be stricter.
            if mkt_score < _env_float("GAP_ATR_MAX_WEAK_MKT_SCORE", 65.0):
                gap_lim = max(0.5, gap_lim - _env_float("GAP_ATR_MAX_WEAK_TIGHTEN", 0.3))
            if gap_atr_max > gap_lim:
                continue

        ext_atr = np.nan
        try:
            c_ser = df["Close"].astype(float)
            ma20_last = safe_float(c_ser.rolling(20).mean().iloc[-1], np.nan)
            atr_abs = safe_float(atr14(df).iloc[-1], np.nan)
            close_last = safe_float(c_ser.iloc[-1], np.nan)
            if np.isfinite(close_last) and np.isfinite(ma20_last) and np.isfinite(atr_abs) and atr_abs > 0:
                ext_atr = float((close_last - ma20_last) / atr_abs)
        except Exception:
            ext_atr = np.nan

        # Simple momentum sanity (helps avoid 'trend setups' inside a longer downtrend)
        ret60 = np.nan
        ret120 = np.nan
        dd60 = np.nan
        up_ratio20 = np.nan
        lr20 = np.nan
        r2_20 = np.nan
        lr60 = np.nan
        r2_60 = np.nan
        uv_ratio20 = np.nan
        dist_days20 = 0
        try:
            c_ser = df["Close"].astype(float)
            if len(c_ser) >= 61:
                ret60 = safe_float((c_ser.iloc[-1] / c_ser.iloc[-61] - 1.0) * 100.0, np.nan)
            if len(c_ser) >= 121:
                ret120 = safe_float((c_ser.iloc[-1] / c_ser.iloc[-121] - 1.0) * 100.0, np.nan)
            sub = c_ser.iloc[-60:]
            if len(sub) >= 20:
                peak = sub.cummax()
                dd60 = safe_float(((sub / peak) - 1.0).min() * 100.0, np.nan)  # negative
            d = c_ser.diff().iloc[-20:]
            if len(d) >= 10:
                up_ratio20 = float((d > 0).mean())

            # Trend smoothness (regression)
            lr20, r2_20 = _linreg_slope_r2(c_ser, 20)
            lr60, r2_60 = _linreg_slope_r2(c_ser, 60)

            # Volume pattern (accumulation / distribution)
            uv_ratio20 = _up_down_volume_ratio(df, 20)
            dist_days20 = _distribution_days(df, 20)
        except Exception:
            pass

        # --- Trend smoothness / strength (orthogonal information) -------------
        er60 = safe_float(efficiency_ratio(df["Close"], 60), np.nan)
        chop14 = safe_float(choppiness_index(df, 14), np.nan)
        adx14v = safe_float(adx(df, 14), np.nan)


        # Hard filters (precision mode): trend smoothness + choppiness.
        # Noise score already penalizes these, but hard filters cut the worst false positives.
        if not is_pos:
            er_base = _env_float("ER60_MIN_BASE", 0.16)
            er_bonus_a1s = _env_float("ER60_MIN_BONUS_A1S", 0.04)
            er_bonus_a1 = _env_float("ER60_MIN_BONUS_A1", 0.02)
            er_bonus_b = _env_float("ER60_MIN_BONUS_B", 0.02)
            er_min = er_base
            if info.setup == "A1-Strong":
                er_min += er_bonus_a1s
            elif info.setup == "A1":
                er_min += er_bonus_a1
            elif info.setup == "B":
                er_min += er_bonus_b

            chop_base = _env_float("CHOP14_MAX_BASE", 68.0)
            chop_bonus_a1s = _env_float("CHOP14_TIGHTEN_A1S", 4.0)
            chop_bonus_a1 = _env_float("CHOP14_TIGHTEN_A1", 2.0)
            chop_bonus_b = _env_float("CHOP14_TIGHTEN_B", 3.0)
            chop_max = chop_base
            if info.setup == "A1-Strong":
                chop_max -= chop_bonus_a1s
            elif info.setup == "A1":
                chop_max -= chop_bonus_a1
            elif info.setup == "B":
                chop_max -= chop_bonus_b

            # In weaker tape, tighten a bit further.
            if mkt_score < _env_float("TREND_QUALITY_WEAK_MKT_SCORE", 65.0):
                er_min += _env_float("TREND_QUALITY_WEAK_ER_ADD", 0.02)
                chop_max -= _env_float("TREND_QUALITY_WEAK_CHOP_SUB", 1.0)

            if np.isfinite(er60) and er60 < er_min:
                continue
            if np.isfinite(chop14) and chop14 > chop_max:
                continue

        # --- RS Line (stock / benchmark): slope + proximity to 60D high --------
        rs_line_pos60 = np.nan  # 1.0 = RS line at 60D high
        rs_line_slope20 = np.nan  # % change in 20D
        try:
            if bench_close is not None:
                b = bench_close.reindex(df.index).astype(float)
                s = df["Close"].astype(float)
                ratio = (s / (b.replace(0.0, np.nan))).dropna()
                if len(ratio) >= 21:
                    rs_line_slope20 = safe_float((ratio.iloc[-1] / ratio.iloc[-21] - 1.0) * 100.0, np.nan)
                if len(ratio) >= 60:
                    hi = float(ratio.iloc[-60:].max())
                else:
                    hi = float(ratio.max()) if len(ratio) else np.nan
                if np.isfinite(hi) and hi > 0 and len(ratio):
                    rs_line_pos60 = safe_float(float(ratio.iloc[-1]) / hi, np.nan)
        except Exception:
            rs_line_pos60 = np.nan
            rs_line_slope20 = np.nan

        # Trend template gate (setup-dependent, market-regime aware)
        trend_min_a1 = _env_float("TREND_MIN_A1", 0.70)
        trend_min_a2 = _env_float("TREND_MIN_A2", 0.62)
        trend_min_b = _env_float("TREND_MIN_B", 0.72)
        trend_min = trend_min_a1
        if info.setup == "A2":
            trend_min = trend_min_a2
        elif info.setup == "B":
            trend_min = trend_min_b
        if mkt_score < _env_float("TREND_WEAK_MKT_SCORE", 65.0):
            trend_min = min(1.0, trend_min + _env_float("TREND_WEAK_BONUS", 0.05))
        if np.isfinite(trend_score) and trend_score < trend_min:
            continue
        if info.setup in ("A1-Strong", "A1") and np.isfinite(ret60) and ret60 < 0:
            continue

        # Relative Strength: composite (20/60/120) vs benchmark.
        # If benchmark is unavailable, fall back to absolute returns.
        b20 = float(topx_ret20) if np.isfinite(topx_ret20) else 0.0
        b60 = float(topx_ret60) if np.isfinite(topx_ret60) else 0.0
        b120 = float(topx_ret120) if np.isfinite(topx_ret120) else 0.0

        rs20 = float("nan")
        rs60 = float("nan")
        rs120 = float("nan")
        rs_comp = float("nan")

        if np.isfinite(r20):
            rs20 = float(r20 - b20)
        if np.isfinite(ret60):
            rs60 = float(ret60 - b60)
        if np.isfinite(ret120):
            rs120 = float(ret120 - b120)

        parts = []
        if np.isfinite(rs20):
            parts.append((0.50, rs20))
        if np.isfinite(rs60):
            parts.append((0.30, rs60))
        if np.isfinite(rs120):
            parts.append((0.20, rs120))
        if parts:
            wsum = sum(w for w, _ in parts)
            rs_comp = float(sum(w * v for w, v in parts) / (wsum or 1.0))

        # Hard exclude on weak RS composite (regime adaptive)
        rs_comp_min = float(rs_comp_min_by_market(mkt_score))
        if np.isfinite(rs_comp) and (rs_comp < rs_comp_min):
            continue

        # Add to cross-sectional pool (for percentile ranking later)
        if np.isfinite(rs_comp):
            rs_pool_syms.append(ticker)
            rs_pool_vals.append(rs_comp)

        # ノイズスコア（イベント起因・滑り地雷の検知）
        #  - 出来高拡大 + ギャップ多発 + ボラ拡大 は「狙える形」(押し目)の期待値を大きく毀損しやすい
        #  - 過剰最適化は避けるため、複合的に悪い場合のみ除外する
        noise_score = 0
        if np.isfinite(vr) and vr >= 1.60:
            noise_score += 1
        if np.isfinite(ac) and ac >= 1.15:
            noise_score += 1
        if np.isfinite(gf) and gf >= 0.20:
            noise_score += 1
        if np.isfinite(rc) and rc >= 1.40:
            noise_score += 1
        # ATR-based overnight gap risk (gap in ATR units)
        if np.isfinite(gap_atr_freq) and gap_atr_freq >= _env_float("GAP_ATR_FREQ_WARN", 0.25):
            noise_score += 1
        # Pullback distribution check: down-volume dominance is a red flag
        if np.isfinite(vol_dry) and vol_dry >= _env_float("VOL_DRY_WARN", 1.35):
            noise_score += 1
        # Overextension (chasing risk)
        if np.isfinite(ext_atr) and ext_atr >= _env_float("EXT_ATR_WARN", 2.8):
            noise_score += 1
        # Deep drawdown within the last ~3 months
        if np.isfinite(dd60) and dd60 <= -20.0:
            noise_score += 1

        # Choppy trend / distribution pressure
        if np.isfinite(r2_60) and (r2_60 <= _env_float("R2_60_WARN", 0.18)):
            noise_score += 1
        if np.isfinite(uv_ratio20) and (uv_ratio20 <= _env_float("UPDN_VOL_WARN", 0.85)):
            noise_score += 1
        if int(dist_days20) >= int(os.getenv("DIST_DAYS_WARN", "4")):
            noise_score += 1

        # Smoothness / chop / trend-strength filters (orthogonal to MA template)
        if np.isfinite(er60) and (er60 <= _env_float("ER60_WARN", 0.20)):
            noise_score += 1
        if np.isfinite(chop14) and (chop14 >= _env_float("CHOP14_WARN", 62.0)):
            noise_score += 1
        if np.isfinite(adx14v) and (adx14v <= _env_float("ADX14_WARN", 14.0)):
            noise_score += 1

        if noise_score >= NOISE_EXCLUDE_SCORE:
            continue

        if np.isfinite(vr):
            if vr <= 0.85:
                quality += 0.05
            elif vr <= 0.95:
                quality += 0.03
            elif vr >= 1.60:
                quality -= 0.08
            elif vr >= 1.30:
                quality -= 0.05

        if np.isfinite(ac):
            if ac <= 0.90:
                quality += 0.05
            elif ac <= 0.98:
                quality += 0.03
            elif ac >= 1.35:
                quality -= 0.08
            elif ac >= 1.15:
                quality -= 0.05

        if np.isfinite(rc):
            # rc>1.0 は直近レンジ拡大。ハンドル/押しが荒い可能性があるので軽く減点。
            if rc >= 1.40:
                quality -= 0.05
            elif rc <= 0.90:
                quality += 0.02

        if np.isfinite(gf):
            if gf >= 0.20:
                quality -= 0.08
            elif gf >= 0.12:
                quality -= 0.05

        if np.isfinite(r20):
            if r20 >= 8.0:
                quality += 0.06
            elif r20 >= 4.0:
                quality += 0.04
            elif r20 >= 0.0:
                quality += 0.02
            elif r20 <= -4.0:
                quality -= 0.05

        # 実行可能性が低い（帯から遠い）ものは、期待値に対して機会損失が出やすい
        if band_dist_pct >= 5.0:
            quality -= 0.04
        elif band_dist_pct >= 3.0:
            quality -= 0.02

        # Relative Strength (composite): prefer leaders.
        if np.isfinite(rs_comp):
            if rs_comp >= 10.0:
                quality += 0.05
            elif rs_comp >= 6.0:
                quality += 0.04
            elif rs_comp >= 2.0:
                quality += 0.02
            elif rs_comp <= -4.0:
                quality -= 0.05

        # Trend smoothness / accumulation proxies
        if np.isfinite(r2_60):
            if r2_60 >= 0.45:
                quality += 0.02
            elif r2_60 <= 0.20:
                quality -= 0.02
        if np.isfinite(uv_ratio20):
            if uv_ratio20 >= 1.25:
                quality += 0.02
            elif uv_ratio20 <= 0.80:
                quality -= 0.02
        if int(dist_days20) >= 4:
            quality -= 0.02

        # --- Smoothness / CHOP / ADX (clean trend preference) -----------------
        if np.isfinite(er60):
            if er60 >= 0.45:
                quality += 0.03
            elif er60 >= 0.35:
                quality += 0.02
            elif er60 <= 0.20:
                quality -= 0.03

        if np.isfinite(chop14):
            if chop14 <= 45.0:
                quality += 0.03
            elif chop14 >= 62.0:
                quality -= 0.03

        if np.isfinite(adx14v):
            if adx14v >= 25.0:
                quality += 0.02
            elif adx14v <= 14.0:
                quality -= 0.02

        # --- RS line: prefer leaders whose RS is at/near highs ----------------
        if np.isfinite(rs_line_pos60):
            if rs_line_pos60 >= 0.985:
                quality += 0.03
            elif rs_line_pos60 <= 0.92:
                quality -= 0.02

        if np.isfinite(rs_line_slope20):
            if rs_line_slope20 >= 3.0:
                quality += 0.02
            elif rs_line_slope20 <= -2.0:
                quality -= 0.02

        # 週足整合（Trueなら加点。Falseはsetup次第で軽い減点）
        if weekly_ok is True:
            quality += 0.03
        elif weekly_ok is False and info.setup not in WEEKLY_STRICT_SETUPS:
            quality -= 0.03

        # band までの距離を ATR で評価（...)
        if np.isfinite(pb_atr):
            if pb_atr >= 1.8:
                quality -= 0.04
            elif pb_atr >= 1.2:
                quality -= 0.02

        # Price impact (Amihud) - penalize thin/impactful names even if ADV is OK
        if np.isfinite(amihud_bps100m):
            if amihud_bps100m <= 25.0:
                quality += 0.02
            elif amihud_bps100m >= 90.0:
                quality -= 0.05
            elif amihud_bps100m >= 60.0:
                quality -= 0.03

        # Trend template / regime (already gated; still helps ranking)
        if np.isfinite(trend_score):
            quality += (trend_score - 0.70) * 0.08

        # BB width contraction: prefer tightening (ratio < 1)
        if np.isfinite(bb_ratio):
            if bb_ratio <= 0.85:
                quality += 0.02
            elif bb_ratio >= 1.15:
                quality -= 0.02

        # Volume dry-up on pullback: prefer down-volume <= up-volume
        if np.isfinite(vol_dry):
            if vol_dry <= 0.85:
                quality += 0.02
            elif vol_dry >= 1.25:
                quality -= 0.02

        # ATR-based gap risk penalty (overnight)
        if np.isfinite(gap_atr_freq) and gap_atr_freq > 0.20:
            quality -= min(0.03, (gap_atr_freq - 0.20) * 0.15)

        # Overextension penalty (chasing risk)
        if np.isfinite(ext_atr) and ext_atr > 2.5:
            quality -= min(0.03, (ext_atr - 2.5) * 0.02)

        # Momentum consistency
        if np.isfinite(dd60) and dd60 < -15.0:
            quality -= 0.02
        if np.isfinite(up_ratio20):
            if up_ratio20 >= 0.58:
                quality += 0.01
            elif up_ratio20 <= 0.42:
                quality -= 0.01

        quality = float(np.clip(quality, -0.30, 0.30))

        # 現値IN判定（運用ルールに沿って“現実的にOK”な条件に限定）
        # - エントリー帯内（微小誤差は許容）
        # - GUではない
        # - Macro警戒ではない
        # - 地合いが一定以上
        # - リスク幅が過大ではない
        # - 到達確率が損益分岐を十分上回る
        band_tol = 0.0005  # 0.05%: 表示丸め/取得誤差の吸収
        in_band = (close_last >= entry_low * (1.0 - band_tol)) and (close_last <= entry_high * (1.0 + band_tol))
        p_hit = float(ev.p_reach)
        rr = float(ev.rr)
        p_be = (1.0 / (rr + 1.0)) if rr > 0 else 1.0
        prob_margin = 0.10
        # 現値INは「事故りやすいノイズ」をさらに排除（世界観：最小事故で積み上げる）
        vr_ok = (not np.isfinite(vr)) or (vr <= 1.35)
        ac_ok = (not np.isfinite(ac)) or (ac <= 1.15)
        gf_ok = (not np.isfinite(gf)) or (gf <= 0.25)
        q_ok = (not np.isfinite(quality)) or (quality >= -0.05)
        noise_ok = (noise_score <= 1)

        # 現値INのリスク判定は「今入るなら」の実効リスクで評価（中央値では事故る）
        risk_now = risk_pct
        if close_last > 0:
            try:
                risk_now = float((close_last - sl) / close_last * 100.0)
            except Exception:
                risk_now = risk_pct

        # --- Market-in (現値IN) gating ---------------------------------------------
        # Prefer market-in only when price is in the *lower/middle* part of the buy band.
        # If price is already near the top edge of the band, it tends to be "chase-y".
        band_pos = float("nan")
        if in_band and (entry_high > entry_low) and close_last > 0:
            band_pos = (close_last - entry_low) / (entry_high - entry_low)
        band_pos_max = safe_float(os.getenv("MARKET_OK_BAND_POS_MAX"), 0.60)  # 0=band bottom, 1=band top
        band_pos_ok = (not np.isfinite(band_pos)) or (band_pos <= band_pos_max)

        market_ok = bool(
            in_band
            and band_pos_ok
            and (not bool(info.gu))
            and (not bool(macro_on))
            and (int(mkt_score) >= 60)
            and (risk_now <= 6.0)
            and (p_hit >= (p_be + prob_margin))
            and vr_ok
            and ac_ok
            and gf_ok
            and q_ok
            and noise_ok
        )
        entry_mode = "MARKET_OK" if market_ok else "LIMIT_ONLY"
        
        cands.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "setup": info.setup,
                "tier": int(info.tier),
                "entry_low": float(entry_low),
                "entry_high": float(entry_high),
                "entry_price": float(entry_price),
                "sl": float(sl),
                "tp1": float(info.tp1),
                "tp2": float(info.tp2),
                "rr": float(ev.rr),
                "struct_ev": float(ev.structural_ev),
                "adj_ev": float(ev.adj_ev),
                "p_hit": float(ev.p_reach),
                "exp_r_hit": float(ev.expected_r * ev.p_reach),
                "cagr": float(ev.cagr_score),
                "expected_days": float(ev.expected_days),
                "rday": float(ev.rday),
                "gu": bool(info.gu),
                "adv20": float(adv),
                "liq_grade": int(liq_grade),
                "liq_adv_min": float(liq_adv_min),
                "mdv20": float(mdv20) if np.isfinite(mdv20) else float("nan"),
                "dv_cv20": float(dv_cv20) if np.isfinite(dv_cv20) else float("nan"),
                "amihud_bps100m": float(amihud_bps100m) if np.isfinite(amihud_bps100m) else float("nan"),
                "atrp": float(atrp),
                "entry_mode": str(entry_mode),
                "band_pos": float(band_pos) if np.isfinite(band_pos) else float("nan"),
                "close_last": float(close_last),
                "risk_pct": float(risk_pct),
                "risk_pct_low": float(risk_pct_low),
                "risk_pct_high": float(risk_pct_high),
                "risk_now": float(risk_now),
                "band_dist": float(band_dist_pct),
                "pb_atr": float(pb_atr) if np.isfinite(pb_atr) else float("nan"),
                "weekly_ok": (None if weekly_ok is None else bool(weekly_ok)),
                "quality": float(quality),
                "vol_ratio": float(vr) if np.isfinite(vr) else float("nan"),
                "atr_contr": float(ac) if np.isfinite(ac) else float("nan"),
                "gap_freq": float(gf) if np.isfinite(gf) else float("nan"),
                "ret20": float(r20) if np.isfinite(r20) else float("nan"),
                "range_contr": float(rc) if np.isfinite(rc) else float("nan"),
                "rs20": float(rs20) if np.isfinite(rs20) else float("nan"),
                "rs60": float(rs60) if np.isfinite(rs60) else float("nan"),
                "rs120": float(rs120) if np.isfinite(rs120) else float("nan"),
                "rs_comp": float(rs_comp) if np.isfinite(rs_comp) else float("nan"),
                "rs_pct": float("nan"),
                "ret120": float(ret120) if np.isfinite(ret120) else float("nan"),
                "lr20": float(lr20) if np.isfinite(lr20) else float("nan"),
                "r2_20": float(r2_20) if np.isfinite(r2_20) else float("nan"),
                "lr60": float(lr60) if np.isfinite(lr60) else float("nan"),
                "r2_60": float(r2_60) if np.isfinite(r2_60) else float("nan"),
                "er60": float(er60) if np.isfinite(er60) else float("nan"),
                "chop14": float(chop14) if np.isfinite(chop14) else float("nan"),
                "adx14": float(adx14v) if np.isfinite(adx14v) else float("nan"),
                "rs_line_pos60": float(rs_line_pos60) if np.isfinite(rs_line_pos60) else float("nan"),
                "rs_line_slope20": float(rs_line_slope20) if np.isfinite(rs_line_slope20) else float("nan"),
                "uv_ratio20": float(uv_ratio20) if np.isfinite(uv_ratio20) else float("nan"),
                "dist_days20": int(dist_days20),
                "noise_score": int(noise_score),
                "trend_score": float(trend_score) if np.isfinite(trend_score) else float("nan"),
                "dist_52w_high": float(dist_52w_high) if np.isfinite(dist_52w_high) else float("nan"),
                "from_52w_low": float(from_52w_low) if np.isfinite(from_52w_low) else float("nan"),
                "bb_ratio": float(bb_ratio) if np.isfinite(bb_ratio) else float("nan"),
                "vol_dry": float(vol_dry) if np.isfinite(vol_dry) else float("nan"),
                "gap_atr_freq": float(gap_atr_freq) if np.isfinite(gap_atr_freq) else float("nan"),
                "gap_atr_max": float(gap_atr_max) if np.isfinite(gap_atr_max) else float("nan"),
                "ext_atr": float(ext_atr) if np.isfinite(ext_atr) else float("nan"),
                "ret60": float(ret60) if np.isfinite(ret60) else float("nan"),
                "dd60": float(dd60) if np.isfinite(dd60) else float("nan"),
                "up_ratio20": float(up_ratio20) if np.isfinite(up_ratio20) else float("nan"),
                "ev_r": float(ev.ev_r),
                "ev_r_day": float(ev.ev_r / max(ev.expected_days, 1e-6)),
                "score": float(ev.cagr_score + (QUALITY_WEIGHT * quality)),
            }
        )

    # --- Liquidity display policy (board-thin avoidance)
    # If any strict-liquidity candidates exist, show only those. This matches the user's preference
    # for thicker boards, even if the list becomes shorter.
    if LIQ_STRICT_ONLY_IF_EXISTS:
        if any(int(c.get("liq_grade", 0)) >= 2 for c in cands):
            cands = [c for c in cands if int(c.get("liq_grade", 0)) >= 2]

    # Latest spec: primary sort is CAGR寄与度（期待R×到達確率）÷想定日数
    # Quality improvements:
    #   - EV_R/day: 損益の符号も含めた期待値効率
    #   - quality: 出来高/ボラ/ギャップ/勢いの軽い補正
    #   - band_dist: エントリー帯から近いほど実行可能性が高い
    cands.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),          # 1) 期待値×品質（上質化の主軸）
            float(x.get("cagr", 0.0)),           # 2) CAGR寄与度(/日)
            float(x.get("ev_r_day", 0.0)),       # 3) 期待値(EV_R)/日
            float(x.get("quality", 0.0)),        # 4) 品質
            -float(x.get("band_dist", 9.9)),     # 5) 帯まで距離(小さいほど)
            float(x.get("p_hit", 0.0)),          # 6) 到達確率
            float(x.get("rr", 0.0)),             # 7) RR(TP1)
            -float(x.get("expected_days", 9.9)), # 8) 想定日数(短いほど)
            -float(x.get("risk_pct", 0.0)),      # 9) リスク幅(小さいほど)
            float(x.get("adv20", 0.0)),          # 10) 流動性
            str(x.get("ticker", "")),           # 11) 安定化
        ),
        reverse=True,
    )
    # ------------------------------------------------------------------
    # Cross-sectional RS percentile (leaders-only bias)
    # ------------------------------------------------------------------
    rs_pct_map: Dict[str, float] = {}
    if len(rs_pool_vals) >= 20:
        try:
            arr = np.asarray(rs_pool_vals, dtype=float)
            order = np.argsort(arr)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(arr))
            denom = float(max(1, len(arr) - 1))
            for sym, rk in zip(rs_pool_syms, ranks):
                rs_pct_map[sym] = float(rk) / denom * 100.0
        except Exception:
            rs_pct_map = {}

    rs_pct_floor = int(os.getenv("RS_PCT_MIN", str(rs_pct_min_by_market(mkt_score))))
    rs_pct_breakout_bonus = int(os.getenv("RS_PCT_BREAKOUT_BONUS", "5"))
    rs_pct_score_w = float(os.getenv("RS_PCT_SCORE_W", "0.10"))

    if rs_pct_map:
        filtered: List[Dict] = []
        for cand in cands:
            sym = cand.get("ticker", "")
            pct = rs_pct_map.get(sym)
            if pct is not None:
                cand["rs_pct"] = round(float(pct), 1)

            # RS-effective: add a small bonus/penalty from RS-line behavior
            rs_eff = float(pct) if pct is not None else 50.0
            try:
                rs_pos = cand.get("rs_line_pos60")
                if rs_pos is not None:
                    rs_pos_f = float(rs_pos)
                    if np.isfinite(rs_pos_f):
                        if rs_pos_f >= 0.985:
                            rs_eff += 3.0
                        elif rs_pos_f >= 0.970:
                            rs_eff += 2.0
                        elif rs_pos_f <= 0.920:
                            rs_eff -= 2.0
                rs_sl = cand.get("rs_line_slope20")
                if rs_sl is not None:
                    rs_sl_f = float(rs_sl)
                    if np.isfinite(rs_sl_f):
                        if rs_sl_f >= 3.0:
                            rs_eff += 1.0
                        elif rs_sl_f <= -2.0:
                            rs_eff -= 1.0
            except Exception:
                pass
            rs_eff = max(1.0, min(99.0, rs_eff))
            cand["rs_eff"] = round(float(rs_eff), 1)

            # Required percentile depends on setup.
            req = rs_pct_floor
            if cand.get("setup") == "B":
                req += rs_pct_breakout_bonus

            # Positions are informational; don't drop them here.
            if cand.get("rowtype") == "POS":
                filtered.append(cand)
                continue

            if float(rs_eff) < req:
                continue

            # Add small leader-bias into final score
            cand["score"] = float(cand.get("score", 0.0)) + rs_pct_score_w * ((float(rs_eff) - 50.0) / 50.0)

            filtered.append(cand)
        cands = filtered

    raw_n = len(cands)

    # diversify
    cands = apply_sector_cap(cands, max_per_sector=2)
    cands = apply_corr_filter(cands, ohlc_map, max_corr=0.75)

    final: List[Dict] = []

    if no_trade:
        # Tier0 exception: max 1, cooldownあり
        if not in_cooldown(state, "tier0_exception_until"):
            tier0 = [c for c in cands if c.get("setup") == "A1-Strong"]
            if tier0:
                pick = tier0[0]
                final = [pick]
                entry_price = float(pick.get("entry_price", (pick["entry_low"] + pick["entry_high"]) / 2.0))
                record_paper_trade(
                    state,
                    bucket="tier0_exception",
                    ticker=pick["ticker"],
                    date_str=today_str,
                    entry=entry_price,
                    sl=pick["sl"],
                    tp2=pick["tp2"],
                    # Align expected_r with realized_r (paper-trade closes at TP2).
                    expected_r=float((float(pick["tp2"]) - float(entry_price)) / max(float(entry_price) - float(pick["sl"]), 1e-9)),
                )
    else:
        final = _apply_setup_mix(cands, max_display(macro_on))

    # Tier2 liquidity cushion
    filtered = []
    for c in final:
        if c.get("tier") == 2 and c.get("setup") in ("A2", "B"):
            if float(c.get("adv20", 0.0)) < 300e6:
                continue
        filtered.append(c)
    final = filtered

    # distortion internal (non-display)
    if not in_cooldown(state, "distortion_until"):
        internal = [c for c in cands if c.get("setup") in ("A1-Strong", "A2")][:2]
        for c in internal:
            entry_price = float(c.get("entry_price", (c["entry_low"] + c["entry_high"]) / 2.0))
            record_paper_trade(
                state,
                bucket="distortion",
                ticker=c["ticker"],
                date_str=today_str,
                entry=entry_price,
                sl=c["sl"],
                tp2=c["tp2"],
                # Align expected_r with realized_r (paper-trade closes at TP2).
                expected_r=float((float(c["tp2"]) - float(entry_price)) / max(float(entry_price) - float(c["sl"]), 1e-9)),
            )

    # Tier0 exception brake
    pt = state.get("paper_trades", {}).get("tier0_exception", [])
    closed = [x for x in pt if x.get("status") == "CLOSED" and x.get("realized_r") is not None]
    if len(closed) >= 4:
        lastN = closed[-4:]
        s = float(np.sum([safe_float(x.get("realized_r"), 0.0) for x in lastN]))
        if s <= -2.0:
            set_cooldown_days(state, "tier0_exception_until", days=4)

    avg_adj = float(np.mean([c["adj_ev"] for c in final])) if final else 0.0
    gu_ratio = float(gu_cnt / max(1, raw_n)) if raw_n > 0 else 0.0

    meta = {
        "raw": int(raw_n),
        "final": int(len(final)),
        "avgAdjEV": float(avg_adj),
        "GU": float(gu_ratio),
        "saucers": scan_saucers(ohlc_map, uni, tcol, max_each=5),
    }

    return final, meta, ohlc_map