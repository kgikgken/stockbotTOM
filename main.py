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
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# JST 日付
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# events 読み込み
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        d = str(r.get("date", "")).strip()
        label = str(r.get("label", "")).strip()
        kind = str(r.get("kind", "")).strip()
        if d and label:
            events.append({"date": d, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    events = load_events()
    warns = []
    for ev in events:
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
            warns.append(f"⚠ {ev['label']}（{when}）: サイズ注意")
    return warns


# ============================================================
# universe 読み込み
# ============================================================
def load_universe(path: str) -> Optional[pd.DataFrame]:
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

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# yfinance
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1)
    return None


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

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else 0.0


def calc_volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < window + 1:
        return 0.03
    v = close.pct_change().rolling(window).std().iloc[-1]
    return float(v) if np.isfinite(v) else 0.03


# ============================================================
# レバ / 立て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 1.8, "強め"
    if mkt_score >= 60:
        return 1.5, "やや強め"
    if mkt_score >= 50:
        return 1.3, "標準"
    if mkt_score >= 40:
        return 1.1, "守り気味"
    return 1.0, "守り"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# dynamic min score
# ============================================================
def dynamic_min_score(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 72.0
    if mkt_score >= 60:
        return 75.0
    if mkt_score >= 50:
        return 78.0
    if mkt_score >= 40:
        return 80.0
    return 82.0


# ============================================================
# セクター強度
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    strength = {}
    for rank, (name, chg) in enumerate(secs[:5]):
        base = 6 - rank
        boost = max(chg, 0.0) * 0.3
        strength[name] = base + boost
    return strength


# ============================================================
# 三階層スコア
# ============================================================
def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    hist: pd.DataFrame,
    score_raw: float,
    mkt_score: int,
    sector_strength: Dict[str, float],
) -> Dict:
    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    rsi = calc_rsi(close)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close)

    quality = float(score_raw)
    setup = 0.0

    if ma5 > ma20 > ma60:
        setup += 12.0

    if 40 <= rsi <= 65:
        setup += 10.0

    if vola20 < 0.02:
        setup += 5.0

    regime = (mkt_score - 50) * 0.12
    if sector_strength:
        regime += sector_strength.get(sector, 0.0)

    total = quality * 0.7 + setup * 1.0 + regime * 0.6

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "score_final": float(total),
        "ma5": ma5,
        "ma20": ma20,
        "atr": atr,
        "vola20": vola20,
        "hist": hist,
    }


# ============================================================
# IN価格
# ============================================================
def compute_entry_price(close: pd.Series, ma5: float, ma20: float, atr: float) -> float:
    price = float(close.iloc[-1])
    target = ma20
    if atr > 0:
        target -= atr * 0.5
    if target > price:
        target = price * 0.995
    return round(float(target), 1)


# ============================================================
# TP/SL
# ============================================================
def calc_candidate_tp_sl(vola20: float, mkt_score: int) -> Tuple[float, float]:
    if vola20 < 0.015:
        tp = 0.06
        sl = -0.03
    elif vola20 < 0.03:
        tp = 0.08
        sl = -0.04
    else:
        tp = 0.12
        sl = -0.06

    if mkt_score >= 70:
        tp += 0.02
    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl


# ============================================================
# 地合い強化
# ============================================================
def enhance_market_score() -> Dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    try:
        sox = yf.Ticker("^SOX").history(period="5d")
        if not sox.empty:
            score += float(np.clip(((sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1) * 100) / 2, -5, 5))
    except Exception:
        pass

    try:
        nvda = yf.Ticker("NVDA").history(period="5d")
        if not nvda.empty:
            score += float(np.clip(((nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1) * 100) / 3, -4, 4))
    except Exception:
        pass

    score = float(np.clip(round(score), 0, 100))
    mkt["score"] = int(score)
    return mkt


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)
    sector_strength = build_sector_strength_map()

    raw: List[Dict] = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue

        if base_score < min_score:
            continue

        raw.append(
            score_candidate(
                ticker, name, sector,
                hist, base_score,
                mkt_score, sector_strength
            )
        )

    raw.sort(key=lambda x: x["score_final"], reverse=True)
    top = raw[:SCREENING_TOP_N]

    final: List[Dict] = []
    for c in top:
        close = c["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, c["ma5"], c["ma20"], c["atr"])
        tp_pct, sl_pct = calc_candidate_tp_sl(c["vola20"], mkt_score)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        price = float(c["price"])
        gap = abs(price - entry) / price if price > 0 else 1.0
        et = "today" if gap <= 0.01 else "soon"

        final.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "sector": c["sector"],
                "score": c["score_final"],
                "price": price,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_type": et,
            }
        )
    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:MAX_FINAL_STOCKS]


# ============================================================
# レポート構築
# ============================================================
def build_report(today_str: str, today_date: datetime.date, mkt: Dict,
                 total_asset: float, pos_text: str) -> str:
    mscore = int(mkt.get("score", 50))
    mcomment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mscore)
    est_asset = total_asset if total_asset > 0 else 2_000_000
    est_asset_i = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join([
            f"{i+1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs)
        ])
    else:
        sec_text = "算出不可（データ不足）"

    ev_lines = build_event_warnings(today_date)
    if not ev_lines:
        ev_lines = ["- 特筆すべきイベントなし（通常モード）"]

    core = run_screening(today_date, mscore)
    today_list = [c for c in core if c["entry_type"] == "today"]
    soon_list = [c for c in core if c["entry_type"] == "soon"]

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mscore}")
    lines.append(f"- コメント: {mcomment}")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 推定運用資産ベース: 約{est_asset_i:,}円")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落）")
    lines.append(sec_text)
    lines.append("")

    lines.append("◆ 今日のイベント・警戒")
    lines.extend(ev_lines)
    lines.append("")

    lines.append(f"◆ Core候補 Aランク（今日IN候補 最大{MAX_FINAL_STOCKS}）")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            lines.append(f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f}")
            lines.append(f"    ・IN: {c['entry']:.1f}")
            lines.append(f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}）")
            lines.append(f"    ・損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）")
            lines.append("")

    lines.append("◆ Core候補 Aランク（数日以内IN候補）")
    if not soon_list:
        lines.append("数日以内INなし")
    else:
        for c in soon_list:
            lines.append(f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f}")
            lines.append(f"    ・IN: {c['entry']:.1f}")
            lines.append(f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}）")
            lines.append(f"    ・損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）")
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    lines.append(f"- MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    lines.append(f"📊 {today_str} ポジション分析")
    lines.append(pos_text.strip())

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return

    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)] or [""]
    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)
            print(ch)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, *_ = analyze_positions(pos_df)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(today_str, today_date, mkt, total_asset, pos_text)

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()