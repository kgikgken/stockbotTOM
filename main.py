# ==========================================
# stockbotTOM main.py（X投稿用140字テンプレ対応版）
# ==========================================

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

HISTORY_PERIOD = "6mo"
MIN_HISTORY_DAYS = 60

MIN_AVG_TURNOVER = 3e8
MAX_ATR_RATIO = 0.06
MAX_ATR_MULTIPLE = 1.8

RSI_MIN = 25
RSI_MAX = 40
MA_TOL = 0.01

VOLUME_SOLDOUT_RATIO = 0.4
VOLUME_SPIKE_RATIO = 2.2

RISK_OFF_THRESHOLD = 40
RISK_ON_THRESHOLD = 60

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

def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def safe_float(x, default=np.nan):
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except:
        return float(default)

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"])
    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)
    return df

def load_earnings():
    if not os.path.exists(EARNINGS_CSV_PATH):
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    df = pd.read_csv(EARNINGS_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    else:
        df["earnings_date"] = pd.NaT
    return df

def load_credit():
    if not os.path.exists(CREDIT_CSV_PATH):
        return pd.DataFrame(columns=["ticker", "margin_ratio", "margin_buy", "margin_sell"])
    df = pd.read_csv(CREDIT_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    for col in ["margin_ratio", "margin_buy", "margin_sell"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    return df

UNIVERSE = load_universe()
EARNINGS_DF = load_earnings()
CREDIT_DF = load_credit()

# ==========================================
# テクニカル指標
# ==========================================

def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df, period=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.rolling(period).mean()
    return df

def enrich_technicals(df):
    df = df.copy()
    df["close"] = df["Close"].astype(float)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["ma75"] = df["close"].rolling(75).mean()
    df = add_rsi(df, 14)
    df = add_atr(df, 14)

    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    df["turnover"] = df["close"] * vol.astype(float)
    return df

def fetch_history(ticker):
    try:
        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", auto_adjust=False, progress=False)
    except:
        return None
    if df is None or df.empty:
        return None
    df = df.tail(120)
    if len(df) < MIN_HISTORY_DAYS:
        return None
    return enrich_technicals(df)

# ==========================================
# ハードフィルター
# ==========================================

def passes_liquidity(df):
    recent = df.tail(20)
    avg_turnover = safe_float(recent["turnover"].mean())
    return np.isfinite(avg_turnover) and avg_turnover >= MIN_AVG_TURNOVER

def passes_volatility(df):
    recent = df.tail(60)
    if recent["atr"].isna().all():
        return False
    last = recent.iloc[-1]
    atr = safe_float(last["atr"])
    close = safe_float(last["close"])
    if atr / close > MAX_ATR_RATIO:
        return False
    atr60 = safe_float(recent["atr"].mean())
    if atr > atr60 * MAX_ATR_MULTIPLE:
        return False
    return True

def passes_trend(df):
    last = df.iloc[-1]
    ma10 = safe_float(last["ma10"])
    ma25 = safe_float(last["ma25"])
    ma75 = safe_float(last["ma75"])
    close = safe_float(last["close"])
    return (ma10 >= ma25 >= ma75) and close >= ma75

def passes_event_risk(ticker, df):
    if EARNINGS_DF.empty:
        return True
    sub = EARNINGS_DF[EARNINGS_DF["ticker"] == ticker]
    if sub.empty:
        return True
    earnings_date = sub["earnings_date"].iloc[0]
    if pd.isna(earnings_date):
        return True
    days = (earnings_date - df.index[-1].date()).days
    return not (-3 <= days <= 3)

def passes_credit_risk(ticker, df):
    if CREDIT_DF.empty:
        return True
    sub = CREDIT_DF[CREDIT_DF["ticker"] == ticker]
    if sub.empty:
        return True
    row = sub.iloc[0]
    ratio = safe_float(row.get("margin_ratio"))
    buy = safe_float(row.get("margin_buy"))
    if np.isfinite(ratio) and ratio > 5:
        return False
    vol_week = safe_float(df["Volume"].tail(5).sum())
    if np.isfinite(buy) and np.isfinite(vol_week) and vol_week > 0:
        if buy / vol_week > 20:
            return False
    return True

# ==========================================
# 出来高サイクル
# ==========================================

def analyze_volume_state(df):
    vol = df["Volume"].astype(float)
    if len(vol) < 30:
        return {"ok": False, "soldout": False, "spike": False}

    v30 = safe_float(vol.tail(30).mean())
    last = safe_float(vol.iloc[-1])

    sold = last < v30 * VOLUME_SOLDOUT_RATIO
    spike = last > v30 * VOLUME_SPIKE_RATIO

    v20 = safe_float(vol.tail(20).mean())
    v10 = safe_float(vol.tail(10).mean())
    v5 = safe_float(vol.tail(5).mean())
    last2 = safe_float(vol.tail(2).mean())

    ok = (v20 > v10 > v5) and (last2 > v5) and not spike

    return {"ok": ok, "soldout": sold, "spike": spike}

# ==========================================
# 押し目判定
# ==========================================

def is_deep_pullback(df):
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])
    if not (np.isfinite(close) and np.isfinite(ma25) and np.isfinite(rsi)):
        return False
    if abs(close - ma25) / ma25 > MA_TOL:
        return False
    return RSI_MIN <= rsi <= RSI_MAX

def analyze_candle(df):
    last = df.iloc[-1]
    o = safe_float(last["Open"])
    h = safe_float(last["High"])
    l = safe_float(last["Low"])
    c = safe_float(last["Close"])
    body = abs(c - o)
    rng = h - l
    lower = (c - l) if c > o else (o - l)
    long_lower = (lower / rng > 0.35) and (lower > body)
    return {"long_lower": long_lower}

# ==========================================
# エントリースコア
# ==========================================

def calc_trend_strength(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ma25 = safe_float(last["ma25"])
    ma25p = safe_float(prev["ma25"])
    ma75 = safe_float(last["ma75"])
    slope = (ma25 - ma25p) / ma25p
    spread = (ma25 - ma75) / ma75
    score = 0
    if slope > 0:
        score += min(10, slope * 2000)
    if spread > 0:
        score += min(10, spread * 50)
    return int(max(0, min(20, score)))

def calc_entry_edge(df, volume_state, candle):
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])

    score = 0
    reasons = []

    if passes_trend(df):
        score += 20
        reasons.append("上昇トレンド継続")

    ts = calc_trend_strength(df)
    if ts > 0:
        score += ts
        reasons.append(f"トレンド強度 +{ts}")

    if np.isfinite(ma25) and ma25 > 0 and np.isfinite(close):
        dist = abs(close - ma25) / ma25
        if dist <= 0.005:
            score += 20
            reasons.append("25MAど真ん中")
        elif dist <= 0.01:
            score += 15
            reasons.append("25MA近辺")

    if np.isfinite(rsi):
        if RSI_MIN <= rsi <= 32:
            score += 25
            reasons.append("RSI深押し")
        elif rsi <= RSI_MAX:
            score += 15
            reasons.append("RSI押し目")

    if volume_state["ok"]:
        score += 20
        reasons.append("出来高反転")
    if volume_state["soldout"]:
        score += 5
        reasons.append("売り枯れ")
    if volume_state["spike"]:
        score -= 20
        reasons.append("出来高スパイク注意")

    if candle["long_lower"]:
        score += 10
        reasons.append("下ヒゲ反転")

    score = int(max(0, min(100, score)))
    return score, reasons

# ==========================================
# 地合い
# ==========================================

def fetch_last_and_change(ticker, period="5d"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        close = df["Close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        return last, ((last / prev - 1) * 100)
    except:
        return np.nan, np.nan

def calc_market_summary():
    lines = []
    score = 50
    dia_last, dia_chg = fetch_last_and_change("DIA")
    qqq_last, qqq_chg = fetch_last_and_change("QQQ")
    soxx_last, soxx_chg = fetch_last_and_change("SOXX")
    iwm_last, iwm_chg = fetch_last_and_change("IWM")
    vix_last, vix_chg = fetch_last_and_change("^VIX")
    tnx_last, tnx_chg = fetch_last_and_change("^TNX")
    usdjpy_last, usdjpy_chg = fetch_last_and_change("JPY=X")

    us_list = [dia_chg, qqq_chg, iwm_chg, soxx_chg]
    us_valid = [x for x in us_list if np.isfinite(x)]
    if us_valid:
        us_avg = sum(us_valid)/len(us_valid)
        score += max(-15, min(15, us_avg * 5))
        lines.append(f"- 米株: DIA {dia_chg:+.1f}% / QQQ {qqq_chg:+.1f}% / SOXX {soxx_chg:+.1f}% / IWM {iwm_chg:+.1f}%")

    if np.isfinite(vix_last):
        if vix_last < 15:
            score += 10
            vtxt = "低ボラ"
        elif vix_last > 25:
            score -= 20
            vtxt = "高ボラ"
        else:
            vtxt = "通常"
        lines.append(f"- VIX {vix_last:.1f}（{vtxt}）")

    if np.isfinite(tnx_last):
        y10 = tnx_last / 10
        if y10 < 4:
            score += 5
        elif y10 > 5:
            score -= 5
        lines.append(f"- 米10年金利 {y10:.2f}%")

    if np.isfinite(usdjpy_last):
        lines.append(f"- ドル円 {usdjpy_last:.1f}")

    score = int(max(0, min(100, score)))

    if score >= RISK_ON_THRESHOLD:
        label = "強め"
        regime = "risk_on"
    elif score <= RISK_OFF_THRESHOLD:
        label = "弱め"
        regime = "risk_off"
    else:
        label = "中立"
        regime = "neutral"

    lines.append(f"→ 地合いスコア {score}（{label}）")

    return {"score": score, "label": label, "lines": lines, "regime": regime}

def calc_theme_score(sector, market):
    base = 50
    r = market["regime"]
    if sector in RISK_SECTORS:
        if r == "risk_on": base += 20
        if r == "risk_off": base -= 10
    if sector in DEFENSIVE_SECTORS:
        if r == "risk_off": base += 15
        if r == "risk_on": base -= 5
    return int(max(0, min(100, base)))

def calc_market_fit(sector, market):
    r = market["regime"]
    if r == "risk_on":
        return 80 if sector in RISK_SECTORS else 60
    if r == "risk_off":
        return 80 if sector in DEFENSIVE_SECTORS else 40
    return 60

def decide_risk_regime_action(market):
    r = market["regime"]
    if r == "risk_on":
        return {"regime_label": "攻め寄り", "max_leverage": 2.0, "max_positions": 4, "comment": "押し目追い風"}
    if r == "risk_off":
        return {"regime_label": "守り優先", "max_positions": 2, "max_leverage": 1.2, "comment": "慎重に"}
    return {"regime_label": "中立", "max_positions": 3, "max_leverage": 1.5, "comment": "軽め"}

# ==========================================
# 売買レベル
# ==========================================

def calc_take_profit(df):
    last = safe_float(df["close"].iloc[-1])
    high10 = safe_float(df["close"].tail(10).max())
    ma10 = safe_float(df["ma10"].iloc[-1])
    return int(high10 * 0.6 + ma10 * 0.4)

def calc_stop_loss(df):
    last = safe_float(df["close"].iloc[-1])
    ma25 = safe_float(df["ma25"].iloc[-1])
    low5 = safe_float(df["close"].tail(5).min())
    return int(min([v for v in [last*0.97, ma25*0.985, low5] if np.isfinite(v)]))

def calc_entry_price(df):
    last = df.iloc[-1]
    vals = [safe_float(last["ma5"]), safe_float(last["ma10"]), safe_float(last["ma25"])]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return 0
    return int(vals[0] * 0.2 + vals[1] * 0.2 + vals[2] * 0.6)

# ==========================================
# スクリーニング
# ==========================================

def classify_core_watch(edge, hard):
    if not hard: return None
    if edge >= 72: return "core"
    if edge >= 60: return "watch"
    return None

def calc_final_rank(edge, theme, fit, market):
    r = market["regime"]
    if r == "risk_on": w1, w2, w3 = 0.65, 0.2, 0.15
    elif r == "risk_off": w1, w2, w3 = 0.50, 0.25, 0.25
    else: w1, w2, w3 = 0.55, 0.25, 0.20
    return edge*w1 + theme*w2 + fit*w3

def screen_candidates(market):
    core_rows = []
    watch_rows = []

    for _, row in UNIVERSE.iterrows():
        t = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(t)
        if df is None: continue
        if not passes_liquidity(df): continue
        if not passes_volatility(df): continue
        if not passes_trend(df): continue
        if not is_deep_pullback(df): continue
        if not passes_event_risk(t, df): continue
        if not passes_credit_risk(t, df): continue

        vol = analyze_volume_state(df)
        candle = analyze_candle(df)
        edge, reasons_edge = calc_entry_edge(df, vol, candle)

        class_type = classify_core_watch(edge, True)
        if class_type is None:
            continue

        theme_score = calc_theme_score(sector, market)
        market_fit = calc_market_fit(sector, market)
        final_rank = calc_final_rank(edge, theme_score, market_fit, market)

        last = df.iloc[-1]
        price = safe_float(last["close"])
        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        ma25 = safe_float(last["ma25"])

        lows = [v for v in [ma5, ma10, ma25] if np.isfinite(v)]
        buy_low = int(min(lows)) if lows else int(price)
        buy_high = int(max(lows)) if lows else int(price)

        tp = calc_take_profit(df)
        sl = calc_stop_loss(df)
        entry_price = calc_entry_price(df)

        reasons = []
        if is_deep_pullback(df): reasons.append("深押し")
        reasons.extend(reasons_edge)
        if vol["soldout"]: reasons.append("売り枯れ")
        if vol["spike"]: reasons.append("出来高スパイク")

        rec = {
            "ticker": t,
            "name": name,
            "sector": sector,
            "class": class_type,
            "entry_edge": edge,
            "theme_score": theme_score,
            "market_fit": market_fit,
            "final_rank": final_rank,
            "price": int(price),
            "buy_low": buy_low,
            "buy_high": buy_high,
            "tp": tp,
            "sl": sl,
            "entry_price": entry_price,
            "reasons": " / ".join(sorted(set(reasons))),
        }

        if class_type == "core":
            core_rows.append(rec)
        else:
            watch_rows.append(rec)

    core_rows = sorted(core_rows, key=lambda x: x["final_rank"], reverse=True)[:4]
    watch_rows = sorted(watch_rows, key=lambda x: x["final_rank"], reverse=True)[:5]

    return core_rows, watch_rows

# ==========================================
# ★ 追加：X投稿用（140字以内）生成
# ==========================================

def build_x_posts(core, watch):
    lines = []
    lines.append("=== X投稿用（140字版） ===")

    def make_core(r):
        t = r["ticker"]
        name = r["name"]
        reason = r["reasons"].split(" / ")[0]
        edge = r["entry_edge"]
        price = r["price"]
        return (
            f"{t} {name}\n"
            f"→ {reason}（Edge {edge}）\n"
            f"現在 {price}円。\n"
            f"気づけるやつだけ見ればいい。"
        )

    def make_watch(r):
        t = r["ticker"]
        name = r["name"]
        reason = r["reasons"].split(" / ")[0]
        edge = r["entry_edge"]
        price = r["price"]
        return (
            f"{t} {name}\n"
            f"→ {reason}（Edge {edge}）\n"
            f"今は仕上がり前。現在 {price}円。\n"
            f"理解できるやつだけ来い。"
        )

    if core:
        lines.append("\n【Core】")
        for r in core:
            lines.append(make_core(r))

    if watch:
        lines.append("\n【Watch】")
        for r in watch:
            lines.append(make_watch(r))

    return "\n".join(lines)

# ==========================================
# メインメッセージ構築
# ==========================================

def build_message():
    today = jst_now().strftime("%Y-%m-%d")

    market = calc_market_summary()
    risk_cfg = decide_risk_regime_action(market)
    core, watch = screen_candidates(market)

    lines = []
    lines.append(f"📅 {today} stockbot TOM 戦略レポート\n")

    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {market['score']}点（{market['label']} / {risk_cfg['regime_label']}）")
    lines.append(f"- レバ上限: 約{risk_cfg['max_leverage']}倍 / 最大{risk_cfg['max_positions']}銘柄")
    lines.append(f"- コメント: {risk_cfg['comment']}\n")

    lines.append("◆ マクロ・地合いサマリー")
    lines.extend(market["lines"])
    lines.append("")

    lines.append("◆ 本命（Core）候補")
    if core:
        for i, r in enumerate(core, 1):
            lines.append(f"{i}. {r['ticker']}（{r['name']}） Edge {r['entry_edge']}")
            lines.append(f"   IN目安 {r['entry_price']}円 / 現在 {r['price']}円")
            lines.append(f"   理由: {r['reasons']}\n")
    else:
        lines.append("- 本日なし\n")

    lines.append("◆ 注目（Watch）候補")
    if watch:
        for i, r in enumerate(watch, 1):
            lines.append(f"{i}. {r['ticker']}（{r['name']}） Edge {r['entry_edge']}")
            lines.append(f"   IN目安 {r['entry_price']}円 / 現在 {r['price']}円\n")
    else:
        lines.append("- 本日なし\n")

    # ★ X投稿文を末尾に追加
    x_posts = build_x_posts(core, watch)
    lines.append(x_posts)

    lines.append("\nOnly the edge. Nothing else.")

    return "\n".join(lines)

# ==========================================
# LINE送信
# ==========================================

def send_line(message):
    token = os.getenv("LINE_TOKEN")
    if not token:
        print(message)
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {"messages": [{"type": "text", "text": message}]}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE status:", resp.status_code)
    except Exception as e:
        print("LINE送信エラー:", e)

# ==========================================
# メイン処理
# ==========================================

def main():
    msg = build_message()
    print(msg)
    send_line(msg)

if __name__ == "__main__":
    main()
