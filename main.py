import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ==========================================
# 基本設定
# ==========================================

UNIVERSE_CSV_PATH = "universe_jpx.csv"
EARNINGS_CSV_PATH = "earnings_jpx.csv"
CREDIT_CSV_PATH = "credit_jpx.csv"

HISTORY_PERIOD = "4mo"   # 重くしすぎない程度に短縮
MIN_HISTORY_DAYS = 60

MIN_AVG_TURNOVER = 3e8   # 3億/日
MAX_ATR_RATIO = 0.06
MAX_ATR_MULTIPLE = 1.8

RSI_MIN = 25
RSI_MAX = 40
MA_TOL_BASE = 0.01       # 通常銘柄の25MA許容 ±1%

VOLUME_SOLDOUT_RATIO = 0.4
VOLUME_SPIKE_STRONG = 2.2   # 強スパイク
VOLUME_SPIKE_WEAK = 1.5     # 軽スパイク

RISK_OFF_THRESHOLD = 40
RISK_ON_THRESHOLD = 60

# 高β銘柄（MA距離の許容を少し広げる）
HIGH_BETA_A = {
    "6920.T",  # レーザーテック
    "8035.T",  # 東エレク
    "6857.T",  # アドバンテスト
    "4063.T",  # 信越化学
    "6723.T",  # ルネサス
    "7735.T",  # スクリーン
    "6146.T",  # ディスコ
}
HIGH_BETA_B = {
    "6976.T",  # 太陽誘電
    "6762.T",  # TDK
    "6758.T",  # ソニーG
    "6954.T",  # ファナック
    "6645.T",  # オムロン
    "6923.T",  # スタンレー
    "6594.T",  # 日本電産
}

# TOPIX-17 セクターETF（Nomura NEXT FUNDS）
TOPIX17_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機器",
    "1623.T": "鉄鋼・非鉄金属",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "IT・サービス他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

DEFENSIVE_SECTORS = [
    "電気・ガス業", "食料品", "医薬品", "陸運業", "空運業",
    "小売業", "サービス業"
]
RISK_SECTORS = [
    "情報・通信業", "電気機器", "機械", "精密機器", "非鉄金属",
    "金属製品", "証券、商品先物取引業", "その他金融業"
]


# ==========================================
# ユーティリティ
# ==========================================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def safe_float(x, default=np.nan) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float(default)


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"])
    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)
    return df


def load_earnings() -> pd.DataFrame:
    if not os.path.exists(EARNINGS_CSV_PATH):
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    df = pd.read_csv(EARNINGS_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date"] = pd.NaT
    return df


def load_credit() -> pd.DataFrame:
    if not os.path.exists(CREDIT_CSV_PATH):
        return pd.DataFrame(
            columns=["ticker", "margin_ratio", "margin_buy", "margin_sell"]
        )
    df = pd.read_csv(CREDIT_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    for col in ["margin_ratio", "margin_buy", "margin_sell"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df


UNIVERSE = load_universe()
EARNINGS_DF = load_earnings()
CREDIT_DF = load_credit()


# ==========================================
# テクニカル指標
# ==========================================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["tr"] = tr
    df["atr"] = tr.rolling(period).mean()
    return df


def enrich_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close"] = df["Close"].astype(float)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["ma75"] = df["close"].rolling(75).mean()
    df = add_rsi(df, period=14)
    df = add_atr(df, period=14)

    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    df["turnover"] = df["close"] * vol.astype(float)

    return df


def fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.tail(100)
    if len(df) < MIN_HISTORY_DAYS:
        return None

    df = enrich_technicals(df)
    return df


# ==========================================
# ハードフィルター
# ==========================================

def passes_liquidity(df: pd.DataFrame) -> bool:
    recent = df.tail(20)
    avg_turnover = safe_float(recent["turnover"].mean())
    if not np.isfinite(avg_turnover):
        return False
    return avg_turnover >= MIN_AVG_TURNOVER


def passes_volatility(df: pd.DataFrame) -> bool:
    recent = df.tail(60).copy()
    if recent["atr"].isna().all():
        return False

    last = recent.iloc[-1]
    atr = safe_float(last["atr"])
    close = safe_float(last["close"])
    if not np.isfinite(atr) or not np.isfinite(close) or close <= 0:
        return False

    atr_ratio = atr / close
    if atr_ratio > MAX_ATR_RATIO:
        return False

    atr60 = safe_float(recent["atr"].mean())
    if np.isfinite(atr60) and atr > atr60 * MAX_ATR_MULTIPLE:
        return False

    return True


def passes_trend(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ma10 = safe_float(last["ma10"])
    ma25 = safe_float(last["ma25"])
    ma75 = safe_float(last["ma75"])
    close = safe_float(last["close"])
    ma25_prev = safe_float(prev["ma25"])

    if not all(np.isfinite(v) for v in [ma10, ma25, ma75, close, ma25_prev]):
        return False

    if not (ma10 >= ma25 >= ma75):
        return False
    if close < ma75:
        return False

    slope = (ma25 - ma25_prev) / ma25_prev
    if slope <= 0:
        return False

    return True


def passes_event_risk(ticker: str, df: pd.DataFrame) -> bool:
    """決算日 ±3営業日を除外。CSVにない銘柄はスルー。"""
    if EARNINGS_DF.empty:
        return True

    sub = EARNINGS_DF[EARNINGS_DF["ticker"] == ticker]
    if sub.empty:
        return True

    earnings_date = sub["earnings_date"].iloc[0]
    if pd.isna(earnings_date):
        return True

    last_date = df.index[-1].date()
    diff = (earnings_date - last_date).days
    if -3 <= diff <= 3:
        return False

    return True


def passes_credit_risk(ticker: str, df: pd.DataFrame) -> bool:
    """
    信用倍率・信用買残の重さから危険銘柄を除外
      - 信用倍率 <= 5
      - 信用買残 / 直近1週間出来高 <= 20
    """
    if CREDIT_DF.empty:
        return True

    sub = CREDIT_DF[CREDIT_DF["ticker"] == ticker]
    if sub.empty:
        return True

    row = sub.iloc[0]
    margin_ratio = safe_float(row.get("margin_ratio", np.nan))
    margin_buy = safe_float(row.get("margin_buy", np.nan))

    if np.isfinite(margin_ratio) and margin_ratio > 5.0:
        return False

    vol_week = safe_float(df["Volume"].tail(5).sum())
    if np.isfinite(margin_buy) and np.isfinite(vol_week) and vol_week > 0:
        buy_vs_vol = margin_buy / vol_week
        if buy_vs_vol > 20.0:
            return False

    return True


# ==========================================
# 出来高サイクル & スパイク
# ==========================================

def analyze_volume_state(df: pd.DataFrame) -> dict:
    vol = df["Volume"].astype(float)
    if len(vol) < 30:
        return {"ok": False, "soldout": False, "weak_spike": False, "strong_spike": False}

    v30 = safe_float(vol.tail(30).mean())
    v20 = safe_float(vol.tail(20).mean())
    v10 = safe_float(vol.tail(10).mean())
    v5 = safe_float(vol.tail(5).mean())
    last2 = safe_float(vol.tail(2).mean())
    last = safe_float(vol.iloc[-1])

    soldout = np.isfinite(v30) and last < v30 * VOLUME_SOLDOUT_RATIO
    weak_spike = np.isfinite(v30) and v5 > v30 * VOLUME_SPIKE_WEAK
    strong_spike = np.isfinite(v30) and last > v30 * VOLUME_SPIKE_STRONG

    cond_cycle = v20 > v10 > v5 and last2 > v5

    ok = bool(cond_cycle and not strong_spike)

    return {
        "ok": ok,
        "soldout": bool(soldout),
        "weak_spike": bool(weak_spike),
        "strong_spike": bool(strong_spike),
    }


# ==========================================
# 押し目判定 & ローソク足
# ==========================================

def get_ma_tolerance_for_ticker(ticker: str) -> float:
    if ticker in HIGH_BETA_A:
        return MA_TOL_BASE + 0.003  # 約1.3%
    if ticker in HIGH_BETA_B:
        return MA_TOL_BASE + 0.001  # 約1.1%
    return MA_TOL_BASE


def is_deep_pullback(df: pd.DataFrame, ticker: str) -> bool:
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])

    if not all(np.isfinite(v) for v in [close, ma25, rsi]):
        return False

    tol = get_ma_tolerance_for_ticker(ticker)
    dist = abs(close - ma25) / ma25 if ma25 != 0 else np.inf
    if dist > tol:
        return False

    if not (RSI_MIN <= rsi <= RSI_MAX):
        return False

    return True


def analyze_candle(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    o = safe_float(last["Open"])
    h = safe_float(last["High"])
    l = safe_float(last["Low"])
    c = safe_float(last["Close"])

    body = abs(c - o)
    range_ = h - l
    lower_shadow = c - l if c > o else o - l

    long_lower = False
    if np.isfinite(range_) and range_ > 0 and np.isfinite(lower_shadow) and np.isfinite(body):
        if (lower_shadow / range_ > 0.35) and (lower_shadow > body):
            long_lower = True

    return {"long_lower": long_lower}


# ==========================================
# エントリースコア（中期用）
# ==========================================

def calc_trend_strength(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ma25 = safe_float(last["ma25"])
    ma25_prev = safe_float(prev["ma25"])
    ma75 = safe_float(last["ma75"])

    if not all(np.isfinite(v) for v in [ma25, ma25_prev, ma75]):
        return 0

    slope = (ma25 - ma25_prev) / ma25_prev
    spread = (ma25 - ma75) / ma75

    score = 0
    if slope > 0:
        score += min(10, slope * 2000)
    if spread > 0:
        score += min(10, spread * 50)

    score = int(max(0, min(20, score)))
    return score


def calc_entry_edge(df: pd.DataFrame, volume_state: dict, candle: dict, ticker: str) -> tuple[int, list[str]]:
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])

    score = 0
    reasons: list[str] = []

    if passes_trend(df):
        score += 20
        reasons.append("上昇トレンド継続")

    trend_strength = calc_trend_strength(df)
    if trend_strength > 0:
        score += trend_strength
        reasons.append(f"トレンド強度+{trend_strength}")

    tol = get_ma_tolerance_for_ticker(ticker)
    dist = np.inf
    if np.isfinite(close) and np.isfinite(ma25) and ma25 > 0:
        dist = abs(close - ma25) / ma25
        if dist <= tol * 0.5:
            score += 20
            reasons.append("25MAど真ん中")
        elif dist <= tol:
            score += 15
            reasons.append("25MA近辺")
        elif dist <= tol * 2:
            score += 5
            reasons.append("25MA圏内")

    if np.isfinite(rsi):
        if RSI_MIN <= rsi <= 32:
            score += 15
            reasons.append("RSI深め")
        elif 32 < rsi <= RSI_MAX:
            score += 5
            reasons.append("RSI軽めの押し目")

    if volume_state["ok"]:
        score += 20
        reasons.append("出来高 減→増の反転")
    if volume_state["soldout"]:
        score += 5
        reasons.append("売り枯れ気味")
    if volume_state["weak_spike"]:
        score -= 15
        reasons.append("出来高ややスパイク")
    if volume_state["strong_spike"]:
        score -= 20
        reasons.append("出来高強スパイク")

    if candle["long_lower"]:
        score += 10
        reasons.append("下ヒゲ反転気味")

    score = int(max(0, min(100, score)))
    return score, sorted(set(reasons))


# ==========================================
# マクロ・地合い（内部利用）
# ==========================================

def fetch_last_and_change(ticker: str, period: str = "5d") -> tuple[float, float]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return np.nan, np.nan

    if df is None or df.empty or "Close" not in df.columns or len(df) < 2:
        return np.nan, np.nan

    close = df["Close"].astype(float)
    last = safe_float(close.iloc[-1])
    prev = safe_float(close.iloc[-2])
    if not np.isfinite(last) or not np.isfinite(prev) or prev == 0:
        return np.nan, np.nan

    chg = (last / prev - 1.0) * 100.0
    return last, chg


def calc_market_summary() -> dict:
    score = 50

    _, dia_chg = fetch_last_and_change("DIA")
    _, qqq_chg = fetch_last_and_change("QQQ")
    _, iwm_chg = fetch_last_and_change("IWM")
    _, soxx_chg = fetch_last_and_change("SOXX")

    vix_last, _ = fetch_last_and_change("^VIX")
    tnx_last, _ = fetch_last_and_change("^TNX")

    us_moves = [dia_chg, qqq_chg, iwm_chg, soxx_chg]
    us_valid = [x for x in us_moves if np.isfinite(x)]
    if us_valid:
        us_avg = sum(us_valid) / len(us_valid)
        score += max(-15, min(15, us_avg * 5))

    if np.isfinite(vix_last):
        if vix_last < 15:
            score += 10
        elif vix_last > 25:
            score -= 20

    if np.isfinite(tnx_last):
        y10 = tnx_last / 10.0
        if y10 < 4.0:
            score += 5
        elif y10 > 5.0:
            score -= 5

    score = int(max(0, min(100, score)))

    if score >= RISK_ON_THRESHOLD:
        label = "やや強め"
        regime = "risk_on"
    elif score <= RISK_OFF_THRESHOLD:
        label = "弱め"
        regime = "risk_off"
    else:
        label = "中立"
        regime = "neutral"

    return {
        "score": score,
        "label": label,
        "regime": regime,
    }


def decide_risk_regime_action(market: dict) -> dict:
    regime = market["regime"]

    if regime == "risk_off":
        return {
            "regime_label": "守り優先",
            "max_leverage": 1.2,
            "max_positions": 2,
            "comment": "新規はかなり厳選。サイズ小さめを基本に。",
        }
    elif regime == "risk_on":
        return {
            "regime_label": "攻め寄り",
            "max_leverage": 2.0,
            "max_positions": 4,
            "comment": "押し目狙い自体は追い風。ただしルール外のINはしない。",
        }
    else:
        return {
            "regime_label": "中立",
            "max_leverage": 1.5,
            "max_positions": 3,
            "comment": "軽めのスイングは可。イベント前に無理なフルベットは不要。",
        }


# ==========================================
# セクター（TOPIX-17 ETF）TOP3
# ==========================================

def fetch_topix17_moves() -> tuple[list[dict], str]:
    results: list[dict] = []

    for etf, name in TOPIX17_ETFS.items():
        try:
            df = yf.download(
                etf,
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            continue
        if df is None or df.empty or len(df) < 2:
            continue
        close = df["Close"].astype(float)
        last = safe_float(close.iloc[-1])
        prev = safe_float(close.iloc[-2])
        if not np.isfinite(last) or not np.isfinite(prev) or prev == 0:
            continue
        chg = (last / prev - 1.0) * 100.0
        results.append(
            {"ticker": etf, "name": name, "last": last, "chg": chg}
        )

    if not results:
        return [], "なし"

    positives = [r for r in results if r["chg"] > 0]
    positives.sort(key=lambda x: x["chg"], reverse=True)
    top3 = positives[:3]

    if not top3:
        return [], "なし"

    top1_name = top3[0]["name"]
    return top3, top1_name


# ==========================================
# テーマ・地合い適合
# ==========================================

def calc_theme_score(sector: str, market: dict) -> int:
    base = 50
    regime = market["regime"]

    if sector in RISK_SECTORS:
        if regime == "risk_on":
            base += 20
        elif regime == "risk_off":
            base -= 10
    if sector in DEFENSIVE_SECTORS:
        if regime == "risk_off":
            base += 15
        elif regime == "risk_on":
            base -= 5

    base = int(max(0, min(100, base)))
    return base


def calc_market_fit(sector: str, market: dict) -> int:
    regime = market["regime"]
    if regime == "risk_on":
        if sector in RISK_SECTORS:
            return 80
        if sector in DEFENSIVE_SECTORS:
            return 60
        return 65
    elif regime == "risk_off":
        if sector in DEFENSIVE_SECTORS:
            return 80
        if sector in RISK_SECTORS:
            return 45
        return 55
    else:
        return 60


def calc_final_rank(entry_edge: int, theme_score: int, market_fit: int, market: dict) -> float:
    regime = market["regime"]
    if regime == "risk_on":
        w_e, w_t, w_m = 0.65, 0.20, 0.15
    elif regime == "risk_off":
        w_e, w_t, w_m = 0.50, 0.25, 0.25
    else:
        w_e, w_t, w_m = 0.55, 0.25, 0.20
    return entry_edge * w_e + theme_score * w_t + market_fit * w_m


# ==========================================
# 売買レベル（中期）
# ==========================================

def calc_take_profit(df: pd.DataFrame) -> int:
    """
    中期スイング用TP
    直近10日高値と10MAをブレンドしたシンプルかつ実戦的な目安。
    """
    last = safe_float(df["close"].iloc[-1])
    recent_high = safe_float(df["close"].tail(10).max())
    ma10 = safe_float(df["ma10"].iloc[-1])

    if not np.isfinite(recent_high) or not np.isfinite(ma10):
        return int(last) if np.isfinite(last) else 0

    tp = recent_high * 0.6 + ma10 * 0.4
    return int(tp)


def calc_stop_loss(df: pd.DataFrame) -> int:
    last = safe_float(df["close"].iloc[-1])
    ma25 = safe_float(df["ma25"].iloc[-1])
    recent_low = safe_float(df["close"].tail(5).min())

    candidates: list[float] = []
    if np.isfinite(recent_low):
        candidates.append(recent_low)
    if np.isfinite(ma25):
        candidates.append(ma25 * 0.985)
    if np.isfinite(last):
        candidates.append(last * 0.97)

    if not candidates:
        return int(last) if np.isfinite(last) else 0

    return int(min(candidates))


def calc_entry_price(df: pd.DataFrame) -> int:
    last = df.iloc[-1]
    ma5 = safe_float(last["ma5"])
    ma10 = safe_float(last["ma10"])
    ma25 = safe_float(last["ma25"])

    vals = [v for v in [ma5, ma10, ma25] if np.isfinite(v)]
    if not vals:
        return 0

    entry = ma5 * 0.2 + ma10 * 0.2 + ma25 * 0.6
    return int(entry)


def calc_shortterm_tp(df: pd.DataFrame) -> int:
    """
    短期TP（1〜3日リバウンド想定）
    TP_short = 直近5日高値×0.5 + 10MA×0.5
    """
    last = df.iloc[-1]
    ma10 = safe_float(last.get("ma10", np.nan))
    recent_high = safe_float(df["close"].tail(5).max())

    if not np.isfinite(recent_high) or not np.isfinite(ma10):
        return 0

    tp_short = recent_high * 0.5 + ma10 * 0.5
    return int(tp_short)


# ==========================================
# 短期パターン検出（ShortTerm）
# ==========================================

def detect_shortterm_patterns(df: pd.DataFrame) -> list[str]:
    patterns: list[str] = []
    if len(df) < 3:
        return patterns

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o = safe_float(last["Open"])
    h = safe_float(last["High"])
    l = safe_float(last["Low"])
    c = safe_float(last["Close"])
    prev_close = safe_float(prev["Close"])
    vol = df["Volume"].astype(float)
    v_last = safe_float(vol.iloc[-1])
    v_prev = safe_float(vol.iloc[-2])
    v20 = safe_float(vol.tail(20).mean())

    ma5 = safe_float(last.get("ma5", np.nan))
    ma10 = safe_float(last.get("ma10", np.nan))
    ma75 = safe_float(last.get("ma75", np.nan))
    rsi = safe_float(last.get("rsi", np.nan))

    # パターン① 下ヒゲ反発
    if all(np.isfinite(v) for v in [o, h, l, c, prev_close, v_last, v_prev, ma75]):
        change = (c - prev_close) / prev_close
        body = c - o
        range_ = h - l
        lower_shadow = (o if c >= o else c) - l
        if (
            change <= -0.03 and          # 実体で-3％以上
            range_ > 0 and
            lower_shadow / range_ >= 0.4 and
            v_last >= v_prev * 2.0 and   # 出来高2倍以上
            c >= ma75                    # 崩壊は除外
        ):
            patterns.append("下ヒゲ反発")

    # パターン② RSIオーバーシュート＋MA5/10
    if all(np.isfinite(v) for v in [ma5, ma10, ma75, c, rsi, v20]):
        # 短期はやや広めに拾う（勝ちやすさと件数のバランス）
        if rsi <= 34:
            dist5 = abs(c - ma5) / ma5 if ma5 != 0 else np.inf
            dist10 = abs(c - ma10) / ma10 if ma10 != 0 else np.inf
            vol_ratio = v_last / v20 if v20 > 0 else 1.0

            if (
                passes_trend(df) and
                (dist5 <= 0.025 or dist10 <= 0.025) and  # MA5/10近辺（±2.5%）
                0.85 <= vol_ratio <= 1.20                # 出来高は平常〜やや増
            ):
                patterns.append("RSIオーバーシュート")

    return sorted(set(patterns))


# ==========================================
# スクリーニング（中期＋短期）
# ==========================================

def classify_core_watch(entry_edge: int) -> str | None:
    if entry_edge >= 75:
        return "core"
    if entry_edge >= 60:
        return "watch"
    return None


def screen_all(market: dict) -> tuple[list[dict], list[dict], list[dict]]:
    core_rows: list[dict] = []
    watch_rows: list[dict] = []
    short_rows: list[dict] = []

    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(ticker)
        if df is None:
            continue

        # 共通のハードフィルタ（短期も中期も最低限は通す）
        if not passes_liquidity(df):
            continue
        if not passes_volatility(df):
            continue
        if not passes_event_risk(ticker, df):
            continue
        if not passes_credit_risk(ticker, df):
            continue

        # ---------- 中期（Core / Watch） ----------
        if passes_trend(df) and is_deep_pullback(df, ticker):
            volume_state = analyze_volume_state(df)
            candle = analyze_candle(df)
            entry_edge, reasons_edge = calc_entry_edge(df, volume_state, candle, ticker)

            class_type = classify_core_watch(entry_edge)
            if class_type is not None:
                theme_score = calc_theme_score(sector, market)
                market_fit = calc_market_fit(sector, market)
                final_rank = calc_final_rank(entry_edge, theme_score, market_fit, market)

                last = df.iloc[-1]
                price = safe_float(last["close"])
                ma5 = safe_float(last["ma5"])
                ma10 = safe_float(last["ma10"])
                ma25 = safe_float(last["ma25"])

                lower_candidates = [v for v in [ma5, ma10, ma25] if np.isfinite(v)]
                if lower_candidates:
                    buy_low = int(min(lower_candidates))
                    buy_high = int(max(lower_candidates))
                else:
                    buy_low = buy_high = int(price)

                tp = calc_take_profit(df)
                sl = calc_stop_loss(df)
                entry_price = calc_entry_price(df)

                rec = {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "class": class_type,
                    "entry_edge": entry_edge,
                    "theme_score": theme_score,
                    "market_fit": market_fit,
                    "final_rank": final_rank,
                    "price": int(price) if np.isfinite(price) else 0,
                    "buy_low": buy_low,
                    "buy_high": buy_high,
                    "tp": tp,
                    "sl": sl,
                    "entry_price": entry_price,
                    "reasons": " / ".join(reasons_edge),
                }

                if class_type == "core":
                    core_rows.append(rec)
                else:
                    watch_rows.append(rec)

        # ---------- 短期（ShortTerm） ----------
        patterns = detect_shortterm_patterns(df)
        if patterns:
            last = df.iloc[-1]
            price = safe_float(last["close"])
            ma5 = safe_float(last.get("ma5", np.nan))
            ma10 = safe_float(last.get("ma10", np.nan))
            atr = safe_float(last.get("atr", np.nan))

            # シンプルなスコア付け（下ヒゲのほうを重く）
            score = 0
            if "下ヒゲ反発" in patterns:
                score += 60
            if "RSIオーバーシュート" in patterns:
                score += 40
            if passes_trend(df):
                score += 10

            # 1〜3日イメージの値幅（MA5/MA10付近＋ATRの1/3）
            if np.isfinite(price) and np.isfinite(ma5) and np.isfinite(ma10) and np.isfinite(atr):
                base_low = min(ma5, ma10)
                base_high = max(ma5, ma10)
                range_low = int(base_low)
                range_high = int(base_high + atr / 3.0)
            elif np.isfinite(price):
                range_low = int(price * 1.01)
                range_high = int(price * 1.03)
            else:
                range_low = range_high = 0

            tp_short = calc_shortterm_tp(df)

            short_rows.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "price": int(price) if np.isfinite(price) else 0,
                    "patterns": " / ".join(patterns),
                    "score": score,
                    "range_low": range_low,
                    "range_high": range_high,
                    "tp_short": tp_short,
                }
            )

    core_rows = sorted(core_rows, key=lambda x: x["final_rank"], reverse=True)[:4]
    watch_rows = sorted(watch_rows, key=lambda x: x["final_rank"], reverse=True)[:5]
    short_rows = sorted(short_rows, key=lambda x: x["score"], reverse=True)[:5]

    return core_rows, watch_rows, short_rows


# ==========================================
# 相場3行まとめ
# ==========================================

def build_three_line_summary(
    market: dict,
    top_sector_name: str,
    core: list[dict],
    watch: list[dict],
    short_rows: list[dict],
) -> list[str]:
    lines: list[str] = []

    regime = market["regime"]
    if regime == "risk_on":
        lines.append("・地合いはやや強め。押し目狙いは前向きに検討。")
    elif regime == "risk_off":
        lines.append("・地合いは弱め。新規は慎重に、サイズ控えめ。")
    else:
        lines.append("・地合いは中立〜レンジ。無理なフルベットは不要。")

    if top_sector_name != "なし":
        lines.append(f"・セクターでは「{top_sector_name}」が相対的に優勢。")
    else:
        lines.append("・セクターは全面的に重く、方向感は出にくい。")

    if core:
        lines.append("・中期はCore中心、短期はShortTerm候補を必要に応じて確認。")
    elif short_rows:
        lines.append("・本命押し目は形成途中。短期リバウンド候補を中心にチェック。")
    else:
        lines.append("・条件が揃うまで待ち優位。ポジション調整と観察がメイン。")

    return lines


# ==========================================
# X投稿用（素材）
# ==========================================

def build_x_templates(
    core: list[dict],
    watch: list[dict],
    short_rows: list[dict],
    market: dict,
) -> str:
    lines: list[str] = []
    lines.append("【X投稿用メモ（stockbotTOM）】")
    lines.append(f"今日の地合い：{market['score']}点 / {market['label']}")
    lines.append("")

    def core_line(r: dict) -> str:
        t = r["ticker"].replace(".T", "")
        name = r["name"]
        edge = r["entry_edge"]
        price = r["price"]
        tp = r["tp"]
        return (
            f"{t} {name}\n"
            f"Edge {edge} / 現{price}円 / TP{tp}円\n"
            f"気づけるやつだけ見ればいい。"
        )

    def watch_line(r: dict) -> str:
        t = r["ticker"].replace(".T", "")
        name = r["name"]
        edge = r["entry_edge"]
        price = r["price"]
        return (
            f"{t} {name}\n"
            f"Edge {edge} / 現{price}円\n"
            f"理解できるやつだけ来い。"
        )

    def short_line(r: dict) -> str:
        t = r["ticker"].replace(".T", "")
        name = r["name"]
        price = r["price"]
        tp_short = r.get("tp_short", 0)
        return (
            f"{t} {name}\n"
            f"短期リバ（1〜3日） / 現{price}円 / TP{tp_short}円\n"
            f"判断できるやつだけ残ればいい。"
        )

    if core:
        lines.append("[Core]")
        for r in core:
            lines.append(core_line(r))
            lines.append("")

    if watch:
        lines.append("[Watch]")
        for r in watch:
            lines.append(watch_line(r))
            lines.append("")

    if short_rows:
        lines.append("[ShortTerm]")
        for r in short_rows:
            lines.append(short_line(r))
            lines.append("")

    if not core and not watch and not short_rows:
        lines.append("今日は条件を満たす銘柄なし。静観メモ。")

    return "\n".join(lines).strip()


# ==========================================
# LINE メッセージ組み立て（5通）
# ==========================================

def build_line_messages() -> list[str]:
    today = jst_now().strftime("%Y-%m-%d")

    market = calc_market_summary()
    risk_cfg = decide_risk_regime_action(market)
    core, watch, short_rows = screen_all(market)
    top3_sectors, top_sector_name = fetch_topix17_moves()

    # ① 今日の結論＋TOP3＋相場3行
    msg1_lines: list[str] = []
    msg1_lines.append(f"📅 {today} stockbotTOM 日報")
    msg1_lines.append("")
    msg1_lines.append("◆ 今日の結論")
    msg1_lines.append(f"- 地合いスコア: {market['score']}点（{market['label']} / {risk_cfg['regime_label']}）")
    msg1_lines.append(
        f"- レバ目安: 最大 約{risk_cfg['max_leverage']:.1f}倍 / ポジ数目安: {risk_cfg['max_positions']}銘柄"
    )
    msg1_lines.append(f"- コメント: {risk_cfg['comment']}")
    msg1_lines.append("")
    msg1_lines.append("◆ 今日のTOPセクター（TOPIX-17）")
    if top3_sectors:
        for i, s in enumerate(top3_sectors, 1):
            msg1_lines.append(f"{i}位: {s['name']}（{s['chg']:+.1f}%）")
    else:
        msg1_lines.append("プラスのセクターなし（全面マイナス）")
    msg1_lines.append("")
    msg1_lines.append("◆ 今日の相場3行まとめ")
    three_lines = build_three_line_summary(market, top_sector_name, core, watch, short_rows)
    msg1_lines.extend(three_lines)
    msg1 = "\n".join(msg1_lines).rstrip()

    # ② Core
    msg2_lines: list[str] = []
    msg2_lines.append("◆ Core（本命候補）")
    if core:
        for i, r in enumerate(core, 1):
            msg2_lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}")
            msg2_lines.append(
                f"   Edge {r['entry_edge']} / Theme {r['theme_score']} / Fit {r['market_fit']}"
            )
            msg2_lines.append(
                f"   IN目安: {r['entry_price']}円 / 現値: {r['price']}円"
            )
            msg2_lines.append(
                f"   TP目安: {r['tp']}円 / SL目安: {r['sl']}円"
            )
            msg2_lines.append("")
    else:
        msg2_lines.append("本命条件を満たす銘柄なし。今日は無理に攻めない選択もあり。")
    msg2 = "\n".join(msg2_lines).rstrip()

    # ③ Watch
    msg3_lines: list[str] = []
    msg3_lines.append("◆ Watch（注目候補）")
    if watch:
        for i, r in enumerate(watch, 1):
            msg3_lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}")
            msg3_lines.append(
                f"   Edge {r['entry_edge']} / IN目安: {r['entry_price']}円 / 現値: {r['price']}円"
            )
            msg3_lines.append("")
    else:
        msg3_lines.append("現時点で条件を満たす注目押し目候補は少ない。")
    msg3 = "\n".join(msg3_lines).rstrip()

    # ④ ShortTerm
    msg4_lines: list[str] = []
    msg4_lines.append("◆ ShortTerm（短期1〜3日候補）")
    if short_rows:
        for i, r in enumerate(short_rows, 1):
            msg4_lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}")
            msg4_lines.append(f"   パターン: {r['patterns']}")
            msg4_lines.append(
                f"   短期イメージ: {r['range_low']}〜{r['range_high']}円 / TP: {r['tp_short']}円 / 現値: {r['price']}円"
            )
            msg4_lines.append("")
    else:
        msg4_lines.append("現在、条件を満たす短期パターン候補はなし。")
    msg4 = "\n".join(msg4_lines).rstrip()

    # ⑤ X投稿用メモ
    x_text = build_x_templates(core, watch, short_rows, market)
    msg5_lines = [x_text, "", "Only the edge. Nothing else."]
    msg5 = "\n".join(msg5_lines).rstrip()

    # ついでにファイルにも書き出しておく（GitHub ActionsのArtifacts用）
    try:
        with open("line_message.txt", "w", encoding="utf-8") as f:
            f.write("\n\n-----\n\n".join([msg1, msg2, msg3, msg4, msg5]))
        with open("x_posts.txt", "w", encoding="utf-8") as f:
            f.write(x_text)
        # screening_result.csv は必要なら別途実装（現状はLINE用に特化）
    except Exception:
        pass

    return [msg1, msg2, msg3, msg4, msg5]


# ==========================================
# LINE 送信
# ==========================================

def send_line(messages: list[str]) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("LINE_TOKEN 未設定のため標準出力へ出力します。")
        for i, m in enumerate(messages, 1):
            print(f"\n===== MESSAGE {i} =====")
            print(m)
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "messages": [{"type": "text", "text": m} for m in messages[:5]]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        print("LINE status:", resp.status_code)
        if resp.status_code != 200:
            print("LINE response:", resp.text)
    except Exception as e:
        print("LINE送信エラー:", e)


# ==========================================
# メイン
# ==========================================

def main() -> None:
    messages = build_line_messages()
    send_line(messages)


if __name__ == "__main__":
    main()
