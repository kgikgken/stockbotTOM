from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 15
MAX_FINAL_STOCKS = 5

EARNINGS_EXCLUDE_DAYS = 3

# リスク管理
MAX_CORE_POSITIONS = 3          # 本命最大
RISK_PER_TRADE = 0.015          # 1.5%/trade
LIQ_MIN_TURNOVER = 100_000_000  # 最低1億円/日

# ギャップ見送り推奨閾値（寄りがINゾーン上限から+1.5%以上なら見送り推奨）
GAP_SKIP_THRESHOLD = 0.015

# テーマブースト（universe_jpx.csv の "theme" 列に入れる想定のラベル）
THEME_BOOST_MAP: Dict[str, float] = {
    # 政策・ディフェンシブ
    "電力": 10.0,
    "防衛": 9.0,
    "インフラ": 8.0,
    "省エネ": 7.0,
    "資源": 7.0,
    # 成長・モメンタム
    "半導体": 9.0,
    "EV": 8.0,
    "電池": 8.0,
    "生成AI": 9.0,
    "クラウド": 7.0,
    # 金融
    "銀行": 7.0,
    "保険": 6.0,
    # その他
    "内需ディフェンシブ": 6.0,
    "リオープン": 5.0,
}


# ============================================================
# 日付 / イベント関連
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    res: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        d = str(r.get("date", "")).strip()
        lbl = str(r.get("label", "")).strip()
        kind = str(r.get("kind", "")).strip()
        if d and lbl:
            res.append({"date": d, "label": lbl, "kind": kind})
    return res


def build_event_warnings(today: datetime.date) -> List[str]:
    evs = load_events()
    res: List[str] = []
    for ev in evs:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        delta = (d - today).days
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            res.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")
    return res


# ============================================================
# Universe / データ取得
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    # theme 列が無くても動くようにする
    if "theme" not in df.columns:
        df["theme"] = ""

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1.0)
    return None


# ============================================================
# テーマ関連ユーティリティ
# ============================================================
def parse_themes(theme_raw: str) -> List[str]:
    """
    "電力,防衛" "電力/政策" "電力・防衛" などを分解してリスト化。
    """
    if not isinstance(theme_raw, str):
        return []
    s = theme_raw.replace("・", ",").replace("/", ",").replace("／", ",").replace("，", ",").replace("、", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def calc_theme_score(themes: List[str]) -> float:
    """
    THEME_BOOST_MAP に基づいて加点。
    複数テーマがある場合は合計。ただし過剰加点を防ぐため上限を設ける。
    """
    if not themes:
        return 0.0
    score = 0.0
    for t in themes:
        score += THEME_BOOST_MAP.get(t, 0.0)
    # 上限（テーマ強くても+18点程度まで）
    return float(np.clip(score, 0.0, 18.0))


# ============================================================
# テクニカル
# ============================================================
def calc_ma(close: pd.Series, w: int) -> float:
    if len(close) < w:
        return float(close.iloc[-1])
    return float(close.rolling(w).mean().iloc[-1])


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) <= period + 1:
        return 50.0
    diff = close.diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = float(rsi.iloc[-1])
    if not np.isfinite(v):
        return 50.0
    return v


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) <= period + 1:
        return 0.0
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if atr is None or not np.isfinite(atr):
        return 0.0
    return float(atr)


def calc_volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < window:
        return 0.03
    ret = close.pct_change(fill_method=None)
    v = ret.rolling(window).std().iloc[-1]
    if v is None or not np.isfinite(v):
        return 0.03
    return float(v)


# ============================================================
# レバ & 建玉
# ============================================================
def recommend_leverage(m: int) -> Tuple[float, str]:
    if m >= 80:
        return 2.0, "攻めMAX"
    if m >= 70:
        return 1.8, "やや攻め"
    if m >= 60:
        return 1.5, "標準〜やや攻め"
    if m >= 50:
        return 1.3, "標準"
    if m >= 40:
        return 1.1, "やや守り"
    return 1.0, "守り"


def calc_max_position(asset: float, lev: float) -> int:
    if not (np.isfinite(asset) and asset > 0 and lev > 0):
        return 0
    return int(round(asset * lev))


# ============================================================
# 地合いで最低スコアライン調整
# ============================================================
def dynamic_min_score(m: int) -> float:
    if m >= 75:
        return 70.0
    if m >= 65:
        return 73.0
    if m >= 55:
        return 76.0
    if m >= 45:
        return 79.0
    return 82.0


# ============================================================
# セクター強度
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    res: Dict[str, float] = {}
    for rank, (name, chg) in enumerate(secs[:5]):
        base = 6 - rank
        boost = max(chg, 0.0) * 0.3
        res[name] = base + boost
    return res


# ============================================================
# スコアウェイト
# ============================================================
def get_score_weights(m: int) -> Tuple[float, float, float]:
    if m >= 75:
        return 0.6, 1.2, 0.7
    if m >= 60:
        return 0.7, 1.0, 0.7
    if m >= 50:
        return 0.8, 0.9, 0.8
    if m >= 40:
        return 0.8, 0.7, 1.0
    return 0.9, 0.6, 1.1


# ============================================================
# 三階層スコア + テーマ
# ============================================================
def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    themes: List[str],
    hist: pd.DataFrame,
    score_raw: float,
    mkt_score: int,
    sector_strength: Dict[str, float],
    theme_score: float,
) -> Dict:
    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close, 20)

    # Quality
    quality_score = float(score_raw)

    # Setup
    setup_score = 0.0

    # トレンド方向
    if ma5 > ma20 > ma60:
        setup_score += 12.0
    elif ma20 > ma5 > ma60:
        setup_score += 6.0
    elif ma20 > ma60 > ma5:
        setup_score += 3.0

    # RSI
    if 40 <= rsi <= 65:
        setup_score += 10.0
    elif 30 <= rsi < 40 or 65 < rsi <= 70:
        setup_score += 3.0
    else:
        setup_score -= 6.0

    # ボラ
    if vola20 < 0.02:
        setup_score += 5.0
    elif vola20 > 0.05:
        setup_score -= 4.0

    # ATRバランス
    if atr and price > 0:
        atr_ratio = atr / price
        if 0.015 <= atr_ratio <= 0.035:
            setup_score += 6.0
        elif atr_ratio < 0.01 or atr_ratio > 0.06:
            setup_score -= 5.0

    # 出来高
    if "Volume" in hist.columns:
        vol = hist["Volume"].astype(float)
        if len(vol) >= 20:
            v_ma = float(vol.rolling(20).mean().iloc[-1])
            v_now = float(vol.iloc[-1])
            if v_ma > 0:
                r = v_now / v_ma
                if r >= 1.5:
                    setup_score += 3.0
                elif r <= 0.5:
                    setup_score -= 3.0

    # 52週位置（高値掴み/ド底回避）
    try:
        hi_52 = float(close.max())
        lo_52 = float(close.min())
        span = hi_52 - lo_52
        if span > 0:
            loc = (price - lo_52) / span  # 0〜1
            if loc > 0.95 and rsi > 65:
                setup_score -= 5.0  # 高値圏での過熱
            elif loc < 0.2 and rsi < 40 and ma20 < ma60:
                setup_score -= 3.0  # ド底逆張りは減点
    except Exception:
        pass

    # Regime + Theme
    regime_score = (mkt_score - 50) * 0.12
    regime_score += sector_strength.get(sector, 0.0)
    regime_score += theme_score  # テーマ加点

    wQ, wS, wR = get_score_weights(mkt_score)
    total = quality_score * wQ + setup_score * wS + regime_score * wR

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "themes": themes,
        "price": price,
        "score_quality": quality_score,
        "score_setup": setup_score,
        "score_regime": regime_score,
        "score_theme": theme_score,
        "score_final": float(total),
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "atr": atr,
        "vola20": vola20,
        "hist": hist,
    }


# ============================================================
# エントリー中心
# ============================================================
def compute_entry_price(close: pd.Series, ma5: float, ma20: float, atr: float) -> float:
    price = float(close.iloc[-1])
    last_low = float(close.iloc[-5:].min())
    target = ma20

    if atr and atr > 0:
        target = target - atr * 0.5

    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3

    if target > price:
        target = price * 0.995

    if target < last_low:
        target = last_low * 1.02

    return round(float(target), 1)


# ============================================================
# エントリーゾーン（AM8戦略向け：狭め＆やや深め）
# ============================================================
def compute_entry_band(entry: float, atr: float, price: float) -> Tuple[float, float]:
    if entry <= 0 or price <= 0:
        return entry, entry

    if not atr or atr <= 0:
        width = entry * 0.007  # ±0.7%
    else:
        width = min(atr * 0.25, entry * 0.007)

    low = max(entry - width, entry * 0.985)   # 最低でも -1.5% まで
    high = min(entry + width, entry * 1.015)  # 最高でも +1.5% まで

    return round(float(low), 1), round(float(high), 1)


# ============================================================
# TP/SL（RRを 2R 付近に）
# ============================================================
def calc_candidate_tp_sl(
    vola20: float,
    mkt_score: int,
    atr_ratio: Optional[float],
    swing_upside: Optional[float],
) -> Tuple[float, float]:
    v = abs(vola20) if np.isfinite(vola20) else 0.03
    ar = abs(atr_ratio) if (atr_ratio is not None and np.isfinite(atr_ratio)) else 0.02

    # ベース骨格：RR高め
    if v < 0.015 and ar < 0.015:
        tp = 0.08   # +8%
        sl = -0.04  # -4%  → 2.0R
    elif v < 0.03 and ar < 0.03:
        tp = 0.10   # +10%
        sl = -0.04  # -4%  → 2.5R
    else:
        tp = 0.14   # +14%
        sl = -0.055 # -5.5%  → 約2.5R

    # 地合い調整
    if mkt_score >= 70:
        tp += 0.02          # 追い風なら利幅伸ばす
    elif mkt_score < 45:
        tp -= 0.02          # 逆風なら控えめ
        sl = max(sl, -0.035)  # 損切りは浅めにタイト化

    # 直近高値までの距離でTP制限
    if swing_upside is not None and swing_upside > 0:
        max_realistic = swing_upside * 0.9
        if tp > max_realistic:
            tp = max(0.07, max_realistic)

    # 安全レンジにクリップ
    tp = float(np.clip(tp, 0.08, 0.22))
    sl = float(np.clip(sl, -0.06, -0.025))

    return tp, sl


# ============================================================
# 地合い補正
# ============================================================
def enhance_market_score() -> Dict:
    base = calc_market_score()
    if isinstance(base, dict):
        score = float(base.get("score", 50))
        comment = str(base.get("comment", ""))
        info = dict(base)
    else:
        score = float(base)
        comment = ""
        info = {"score": int(score), "comment": comment}

    score = float(np.clip(score, 0, 100))

    # 日経
    try:
        nikkei = yf.Ticker("^N225").history(period="6d")
        if nikkei is not None and not nikkei.empty and len(nikkei) >= 2:
            n_chg = float(nikkei["Close"].iloc[-1] / nikkei["Close"].iloc[0] - 1.0) * 100
            score += float(np.clip(n_chg / 2.5, -6, 6))
    except Exception:
        pass

    # SOX
    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100
            score += float(np.clip(sox_chg / 3.0, -5, 5))
    except Exception:
        pass

    # NVDA
    try:
        nv = yf.Ticker("NVDA").history(period="6d")
        if nv is not None and not nv.empty and len(nv) >= 2:
            nv_chg = float(nv["Close"].iloc[-1] / nv["Close"].iloc[0] - 1.0) * 100
            score += float(np.clip(nv_chg / 4.0, -4, 4))
    except Exception:
        pass

    # FX
    try:
        fx = yf.Ticker("JPY=X").history(period="6d")
        if fx is not None and not fx.empty and len(fx) >= 2:
            fx_chg = float(fx["Close"].iloc[-1] / fx["Close"].iloc[0] - 1.0) * 100
            score += float(np.clip(fx_chg / 4.0, -3, 3))
    except Exception:
        pass

    score = float(np.clip(round(score), 0, 100))
    info["score"] = int(score)
    if not info.get("comment"):
        if score >= 70:
            info["comment"] = "リスクオン寄り（押し目＋強いテーマに資金集中）"
        elif score >= 50:
            info["comment"] = "中立〜やや追い風（本命押し目のみ厳選）"
        elif score >= 40:
            info["comment"] = "やや逆風（ロット控えめ、ポジション数も絞る）"
        else:
            info["comment"] = "リスクオフ気味（基本は様子見〜縮小）"
    return info


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime.date, mkt_score: int, total_asset: float) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)
    sector_strength = build_sector_strength_map()

    raw_candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        theme_raw = str(row.get("theme", "")).strip()
        themes = parse_themes(theme_raw)
        theme_score = calc_theme_score(themes)

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        # 流動性フィルタ
        try:
            close = hist["Close"].astype(float)
            vol = hist["Volume"].astype(float)
            turnover = close * vol
            if len(turnover) < 20:
                continue
            avg_turnover_20 = float(turnover.rolling(20).mean().iloc[-1])
            if avg_turnover_20 < LIQ_MIN_TURNOVER:
                continue
        except Exception:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue
        if base_score < min_score:
            continue

        info = score_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            themes=themes,
            hist=hist,
            score_raw=base_score,
            mkt_score=mkt_score,
            sector_strength=sector_strength,
            theme_score=theme_score,
        )
        raw_candidates.append(info)

    raw_candidates.sort(key=lambda x: x["score_final"], reverse=True)
    topN = raw_candidates[:SCREENING_TOP_N]

    final_list: List[Dict] = []
    risk_amount = float(total_asset) * RISK_PER_TRADE

    for c in topN:
        close = c["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, c["ma5"], c["ma20"], c["atr"])
        price = float(c["price"])
        atr = float(c["atr"]) if c["atr"] is not None else 0.0
        vola20 = c["vola20"]

        atr_ratio = (c["atr"] / price) if (price > 0 and c["atr"] is not None and price > 0) else None

        if len(close) >= 20 and entry > 0:
            swing_high = float(close.tail(20).max())
            swing_upside = (swing_high / entry - 1.0) if swing_high > entry else None
        else:
            swing_upside = None

        tp_pct, sl_pct = calc_candidate_tp_sl(vola20, mkt_score, atr_ratio, swing_upside)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        rr = tp_pct / abs(sl_pct) if sl_pct < 0 else np.nan

        # 想定ホールド（日数ラベル）
        if vola20 < 0.015:
            hold_days = "7〜12日"
        elif vola20 < 0.03:
            hold_days = "4〜8日"
        else:
            hold_days = "2〜5日"

        entry_low, entry_high = compute_entry_band(entry, atr, price)

        pos_shares = 0
        pos_yen = 0.0
        loss_yen = 0.0
        gain_yen = 0.0
        if entry > 0 and sl_pct < 0:
            per_share_risk = entry * abs(sl_pct)
            if per_share_risk > 0:
                raw = int(risk_amount // per_share_risk)
                pos_shares = (raw // 100) * 100
                if pos_shares > 0:
                    pos_yen = pos_shares * entry
                    loss_yen = pos_shares * entry * abs(sl_pct)
                    gain_yen = pos_shares * entry * tp_pct

        price_now = float(c["price"])
        gap_ratio = abs(price_now - entry) / price_now if price_now > 0 else 1.0
        entry_type = "today" if gap_ratio <= 0.01 else "soon"

        final_list.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "sector": c["sector"],
                "themes": c["themes"],
                "score": c["score_final"],
                "price": price_now,
                "entry": entry,
                "entry_low": entry_low,
                "entry_high": entry_high,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "rr": rr,
                "hold_days": hold_days,
                "pos_shares": pos_shares,
                "pos_yen": pos_yen,
                "loss_yen": loss_yen,
                "gain_yen": gain_yen,
                "entry_type": entry_type,
            }
        )

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


# ============================================================
# ログ出力（検証用）
# ============================================================
def save_screening_log(today_date: datetime.date, mkt_score: int, core_list: List[Dict]) -> None:
    try:
        if not core_list:
            return
        rows = []
        for c in core_list:
            rows.append(
                {
                    "date": today_date.isoformat(),
                    "mkt_score": mkt_score,
                    "ticker": c["ticker"],
                    "name": c["name"],
                    "sector": c["sector"],
                    "themes": "|".join(c.get("themes", [])),
                    "score": c["score"],
                    "price": c["price"],
                    "entry": c["entry"],
                    "entry_low": c["entry_low"],
                    "entry_high": c["entry_high"],
                    "tp_pct": c["tp_pct"],
                    "sl_pct": c["sl_pct"],
                    "tp_price": c["tp_price"],
                    "sl_price": c["sl_price"],
                    "rr": c["rr"],
                    "hold_days": c["hold_days"],
                    "pos_shares": c["pos_shares"],
                    "pos_yen": c["pos_yen"],
                    "loss_yen": c["loss_yen"],
                    "gain_yen": c["gain_yen"],
                    "entry_type": c["entry_type"],
                }
            )
        os.makedirs("logs", exist_ok=True)
        fname = os.path.join("logs", f"screening_{today_date.strftime('%Y%m%d')}.csv")
        df_log = pd.DataFrame(rows)
        df_log.to_csv(fname, index=False)
    except Exception as e:
        print("[WARN] failed to save screening log:", e)


# ============================================================
# レポート
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mkt_score)

    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    secs = top_sectors_5d()
    if secs:
        sec_lines = [f"{i+1}. {n} ({chg:+.2f}%)" for i, (n, chg) in enumerate(secs)]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可"

    ev_lines = build_event_warnings(today_date)
    if not ev_lines:
        ev_lines = ["- 特筆すべきイベントなし（通常）"]

    if any(line.startswith("⚠") for line in ev_lines):
        if rec_lev > 1.3:
            rec_lev = 1.3
            lev_comment += " / 重要イベントのため本日のレバ上限1.3倍"

    max_pos = calc_max_position(est_asset, rec_lev)

    core_list = run_screening(today_date, mkt_score, est_asset)
    today_list = [c for c in core_list if c["entry_type"] == "today"]
    soon_list = [c for c in core_list if c["entry_type"] == "soon"]

    # ログ保存（あとで検証するため）
    save_screening_log(today_date, mkt_score, core_list)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 運用資産想定: 約{est_asset_int:,}円")
    lines.append(f"- 同時最大本命銘柄数: {MAX_CORE_POSITIONS}銘柄")
    lines.append("")
    lines.append("※寄り付きが INゾーン上限より +1.5%以上高い場合は、その日は見送り推奨")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落）")
    lines.append(sec_text)
    lines.append("")

    lines.append("◆ 今日のイベント・警戒")
    for ev in ev_lines:
        lines.append(ev)
    lines.append("")

    lines.append(f"◆ Core候補 Aランク（今日IN候補 最大{MAX_FINAL_STOCKS}）")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            theme_str = ", ".join(c.get("themes", [])) if c.get("themes") else "-"
            lines.append(
                f"- {c['ticker']} {c['name']} "
                f"Score:{c['score']:.1f} 現値:{c['price']:.1f} "
                f"[{c['sector']} / テーマ:{theme_str}]"
            )
            lines.append(
                f"    ・INゾーン: {c['entry_low']:.1f}〜{c['entry_high']:.1f}（中心{c['entry']:.1f}）"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}） 損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}） RR:{c['rr']:.1f}R"
            )
            lines.append(
                f"    ・想定ホールド: {c['hold_days']}"
            )
            if c["pos_shares"] > 0:
                lines.append(
                    f"    ・推奨: {c['pos_shares']}株 ≒{int(c['pos_yen']):,}円 / "
                    f"損失~{int(c['loss_yen']):,}円 利確~{int(c['gain_yen']):,}円"
                )
            lines.append("")

    lines.append("◆ Core候補 Aランク（数日以内IN）")
    if not soon_list:
        lines.append("数日以内IN候補なし")
    else:
        for c in soon_list:
            theme_str = ", ".join(c.get("themes", [])) if c.get("themes") else "-"
            lines.append(
                f"- {c['ticker']} {c['name']} "
                f"Score:{c['score']:.1f} 現値:{c['price']:.1f} "
                f"[{c['sector']} / テーマ:{theme_str}]"
            )
            lines.append(
                f"    ・理想IN:{c['entry']:.1f} ゾーン:{c['entry_low']:.1f}〜{c['entry_high']:.1f}"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}% 損切:{c['sl_pct']*100:.1f}% RR:{c['rr']:.1f}R"
            )
            lines.append(
                f"    ・想定ホールド: {c['hold_days']}"
            )
            if c["pos_shares"] > 0:
                lines.append(
                    f"    ・推奨:{c['pos_shares']}株 ≒{int(c['pos_yen']):,}円 / "
                    f"損失~{int(c['loss_yen']):,}円 利確~{int(c['gain_yen']):,}円"
                )
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍 / MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append(pos_text.strip())

    long_report = "\n".join(lines)

    short_lines: List[str] = []
    short_lines.append(f"📅 {today_str} stockbotTOM 要約")
    short_lines.append(f"- 地合い:{mkt_score} / レバ:{rec_lev:.1f}倍")
    if core_list:
        best = core_list[0]
        theme_str = ", ".join(best.get("themes", [])) if best.get("themes") else "-"
        short_lines.append(
            f"- 本命: {best['ticker']} {best['name']} "
            f"Score:{best['score']:.1f} [{best['sector']} / テーマ:{theme_str}]"
        )
        short_lines.append(
            f"  IN:{best['entry']:.1f} RR:{best['rr']:.1f}R "
            f"ロット:{best['pos_shares']}株 想定:{best['hold_days']}"
        )
    else:
        short_lines.append("- 本命なし（今日は無理に攻めない日）")
    short_lines.append(f"- MAX建て玉: 約{max_pos:,}円")

    short_report = "\n".join(short_lines)

    return long_report + "\n\n-----\n\n" + short_report


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return
    chunk = 3900
    parts = [text[i:i + chunk] for i in range(0, len(text), chunk)] or [""]
    for ch in parts:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE]", r.status_code, r.text)
        except Exception as e:
            print("[LINE ERROR]", e)
            print(ch)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, _, _, _ = analyze_positions(pos_df)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    rep = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
    )

    print(rep)
    send_line(rep)


if __name__ == "__main__":
    main()
