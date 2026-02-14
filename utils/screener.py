from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.util import download_history_bulk, safe_float, is_abnormal_stock
from utils.setup import build_setup_info, liquidity_filters
from utils.rr_ev import calc_ev, pass_thresholds
from utils.diversify import apply_sector_cap, apply_corr_filter
from utils.screen_logic import no_trade_conditions, max_display
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
# - adv20: mean traded value over last 20 sessions (¥)
# - mdv20: median traded value over last 20 sessions (¥) to avoid spike-driven illusions
# Primary intent: avoid slippage/partial fills on thin boards.
ADV_MIN_MAIN = 500e6          # default minimum ADV for main candidates (¥/day)
ADV_MIN_A1_STRONG = 350e6     # allow slightly lower ADV for top-tier setups
MDV_FACTOR = 0.70             # mdv20 must be at least adv_min * factor
DV_CV_MAX = 2.0               # if traded value CV is too large (spiky), exclude (unless very liquid)
AMIHUD_MAX_BPS100M = 120.0     # Amihud proxy threshold (bps per 1億円 of traded value)


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
    tickers = uni[tcol].astype(str).tolist()
    ohlc_map = download_history_bulk(tickers, period="780d", auto_adjust=True, group_size=200)

    # --- Index telemetry (for relative strength / RS)
    # Keep it lightweight: one index series, 20/60d relative strength.
    topx_ret20 = float("nan")
    topx_ret60 = float("nan")
    try:
        idx_map = download_history_bulk(["^TOPX"], period="260d", auto_adjust=True, group_size=1)
        topx = idx_map.get("^TOPX")
        if topx is not None and not topx.empty and "Close" in topx.columns:
            topx_ret20 = _ret_n(topx["Close"].astype(float), 20)
            topx_ret60 = _ret_n(topx["Close"].astype(float), 60)
    except Exception:
        # if index fetch fails, RS features are simply disabled
        topx_ret20 = float("nan")
        topx_ret60 = float("nan")

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
        # Use traded value (Close*Volume) statistics to avoid slippage mines.
        adv_min = float(ADV_MIN_A1_STRONG if info.setup == "A1-Strong" else ADV_MIN_MAIN)
        if not (np.isfinite(adv) and float(adv) >= adv_min):
            continue

        dv20 = (df["Close"].astype(float) * df["Volume"].astype(float)).tail(20).dropna()
        mdv20 = safe_float(dv20.median(), np.nan)
        dv_cv20 = safe_float((dv20.std() / (dv20.mean() + 1e-9)), np.nan)
        if not (np.isfinite(mdv20) and float(mdv20) >= (adv_min * MDV_FACTOR)):
            continue
        # If traded value is extremely spiky, treat it as unreliable liquidity unless it is very liquid.
        if np.isfinite(dv_cv20) and float(dv_cv20) > DV_CV_MAX and float(adv) < (adv_min * 2.0):
            continue

        # Price impact proxy (Amihud illiquidity)
        # approx: mean(|ret| / traded_value). We scale it into bps per 1億円 so it's interpretable.
        amihud_bps100m = float('nan')
        try:
            ret_abs20 = df['Close'].astype(float).pct_change(fill_method=None).tail(20).abs()
            ill = (ret_abs20 / (dv20 + 1e-9)).replace([np.inf, -np.inf], np.nan)
            amihud = safe_float(ill.mean(), np.nan)
            if np.isfinite(amihud):
                amihud_bps100m = float(amihud * 1e8 * 10000.0)
        except Exception:
            amihud_bps100m = float('nan')
        if np.isfinite(amihud_bps100m) and float(amihud_bps100m) > AMIHUD_MAX_BPS100M:
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

        # RS20（TOPIX比）：指数より強いものを上に。
        rs20 = float("nan")
        if np.isfinite(r20) and np.isfinite(topx_ret20):
            rs20 = float(r20 - float(topx_ret20))
            if rs20 <= RS20_EXCLUDE:
                continue

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

        # RS20（TOPIX比）: 「指数に勝つ」銘柄を優先
        #  - r20 だけだと相場全体が上げている局面で見誤るので、相対で補正
        if np.isfinite(rs20):
            if rs20 >= 10.0:
                quality += 0.05
            elif rs20 >= 6.0:
                quality += 0.04
            elif rs20 >= 2.0:
                quality += 0.02
            elif rs20 <= -4.0:
                quality -= 0.05

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

        quality = float(np.clip(quality, -0.25, 0.25))

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

        market_ok = bool(
            in_band
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
                "mdv20": float(mdv20) if np.isfinite(mdv20) else float("nan"),
                "dv_cv20": float(dv_cv20) if np.isfinite(dv_cv20) else float("nan"),
                "amihud_bps100m": float(amihud_bps100m) if np.isfinite(amihud_bps100m) else float("nan"),
                "atrp": float(atrp),
                "entry_mode": str(entry_mode),
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
                "noise_score": int(noise_score),
                "ev_r": float(ev.ev_r),
                "ev_r_day": float(ev.ev_r / max(ev.expected_days, 1e-6)),
                "score": float(ev.cagr_score + (QUALITY_WEIGHT * quality)),
            }
        )

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