
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
EVENTS_PATH = "events.csv"      # あれば読む（無ければ無視）
WORKER_URL = os.getenv("WORKER_URL")

# スクリーニング関連
SCREENING_TOP_N = 15           # まずは Top15 まで抽出
MAX_FINAL_STOCKS = 5           # 最終的に LINE に出すのは最大5銘柄

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3

# リスク管理関連
MAX_CORE_POSITIONS = 3          # 同時最大本命銘柄数
RISK_PER_TRADE = 0.015          # Core: 1.5%/trade
LIQ_MIN_TURNOVER = 100_000_000  # 最低1億円/日（将来の流動性フィルタ用）

# Core 固定 TP/SL（RR=2.0）
CORE_TP_PCT = 0.08
CORE_SL_PCT = -0.04

# IN ゾーン幅（中心 ±0.5%）
IN_ZONE_HALF_PCT = 0.005


# ============================================================
# 日付 / イベント関連
# ============================================================
def jst_today_date() -> datetime.date:
    """JST の「今日」の date を返す"""
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    """
    events.csv があれば読み込んで
    [{"date": "2025-12-13", "label": "FOMC", "kind": "macro"}, ...] を返す。
    無ければ空リスト。
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read events: {e}")
        return []

    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        date_str = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    """
    events.csv ベースでイベント警戒文言を作る。
    イベントの2日前〜翌日まで警告。
    """
    events = load_events()
    warns: List[str] = []
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
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")
    return warns


# ============================================================
# Universe / データ取得
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    if "ticker" not in df.columns:
        print("[WARN] universe has no 'ticker' column")
        return None

    df["ticker"] = df["ticker"].astype(str)

    # earnings_date を一度だけパース
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    """決算日 ±EARNINGS_EXCLUDE_DAYS に入っていれば True"""
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    """
    株価履歴取得（簡易リトライ付き）
    yfinance 側の一時エラー時に 1 回だけ待って再試行。
    """
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] fetch history failed {ticker} (try {attempt+1}): {e}")
            time.sleep(1.0)

    return None


# ============================================================
# テクニカル指標
# ============================================================
def calc_ma(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return float(close.iloc[-1])
    return float(close.rolling(window).mean().iloc[-1])


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
    if len(close) < window + 1:
        return 0.03

    ret = close.pct_change(fill_method=None)
    v = ret.rolling(window).std().iloc[-1]

    if v is None or not np.isfinite(v):
        return 0.03
    return float(v)


# ============================================================
# レバレッジ / 建て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    """
    地合いスコアから推奨レバ / コメント
    “1年後の資産最大化” を意識して、
    強いときは少し攻め、弱いときはロットを絞る。
    """
    if mkt_score >= 80:
        return 2.5, "攻めMAX（ただしルール外エントリー禁止）"
    if mkt_score >= 70:
        return 2.0, "やや攻め（押し目＋強いブレイク）"
    if mkt_score >= 60:
        return 1.5, "標準〜やや攻め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "標準（本命押し目のみ）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規は最小ロット〜様子見）"


def calc_max_position(total_asset: float, lev: float) -> int:
    """今日使っていい建て玉最大金額"""
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# 動的な最低スコアライン（地合い連動）
# ============================================================
def dynamic_min_score(mkt_score: int) -> float:
    """
    地合いが強いほど「少し緩く」、弱いほど「厳しく」フィルタする。
    """
    if mkt_score >= 75:
        return 70.0
    if mkt_score >= 65:
        return 73.0
    if mkt_score >= 55:
        return 76.0
    if mkt_score >= 45:
        return 79.0
    return 82.0


# ============================================================
# セクター強度（Top5をブースト）
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    """
    top_sectors_5d() をスコア化して銘柄スコアに加点する。
    上位ほど、上昇率が高いほどブースト。
    """
    secs = top_sectors_5d()
    strength: Dict[str, float] = {}

    for rank, (name, chg) in enumerate(secs[:5]):
        base = 6 - rank  # 1位:6, 2位:5, ...
        boost = max(chg, 0.0) * 0.3
        strength[name] = base + boost

    return strength


# ============================================================
# 三階層スコアの重み（地合いで可変）
# ============================================================
def get_score_weights(mkt_score: int) -> Tuple[float, float, float]:
    """
    Quality / Setup / Regime の重みを地合いで変える。
    強いときは Setup（チャート形状）を重視、
    弱いときは Regime（地合い・セクター）を重視。
    """
    if mkt_score >= 75:
        # 強いトレンド相場：形が良ければ伸ばす
        return 0.6, 1.2, 0.7
    if mkt_score >= 60:
        # 通常〜やや追い風
        return 0.7, 1.0, 0.7
    if mkt_score >= 50:
        # 中立〜やや逆風：Quality 少し重視
        return 0.8, 0.9, 0.8
    if mkt_score >= 40:
        # 弱い地合い：Regime をより重視
        return 0.8, 0.7, 1.0
    # 壊れ気味の地合い：Regime を最重視
    return 0.9, 0.6, 1.1


# ============================================================
# Top候補用の三階層スコアリング
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
    """
    Quality / Setup / Regime の三階層でスコアを構成し、
    “今日から3〜10日スイングで勝ちやすいか” を判定する。
    """

    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close, 20)

    # --- Quality（ベースは ACDE） ---
    quality_score = float(score_raw)

    # --- Setup（短期の形・テクニカル） ---
    setup_score = 0.0

    # 1. トレンド方向（MAの並び）
    if ma5 > ma20 > ma60:
        setup_score += 12.0
    elif ma20 > ma5 > ma60:
        setup_score += 6.0
    elif ma20 > ma60 > ma5:
        setup_score += 3.0

    # 2. RSI（過熱 / 売られ過ぎの調整）
    if 40 <= rsi <= 65:
        setup_score += 10.0
    elif 30 <= rsi < 40 or 65 < rsi <= 70:
        setup_score += 3.0
    else:
        setup_score -= 6.0

    # 3. ボラティリティの安定感
    if vola20 < 0.02:
        setup_score += 5.0
    elif vola20 > 0.05:
        setup_score -= 4.0

    # 4. ATR（値幅の取りやすさ）
    if atr and price > 0:
        atr_ratio = atr / price
        if 0.015 <= atr_ratio <= 0.035:
            setup_score += 6.0
        elif atr_ratio < 0.01 or atr_ratio > 0.06:
            setup_score -= 5.0

    # 5. 出来高（薄商いを減点）
    if "Volume" in hist.columns:
        vol = hist["Volume"].astype(float)
        if len(vol) >= 20:
            v_ma = float(vol.rolling(20).mean().iloc[-1])
            v_now = float(vol.iloc[-1])
            if v_ma > 0:
                ratio = v_now / v_ma
                if ratio >= 1.5:
                    setup_score += 3.0
                elif ratio <= 0.5:
                    setup_score -= 3.0

    # --- Regime（地合い・セクター） ---
    regime_score = 0.0

    # 地合い：50を中立として上下にオフセット
    regime_score += (mkt_score - 50) * 0.12

    # セクター強度ブースト
    if sector_strength:
        regime_score += sector_strength.get(sector, 0.0)

    # --- 三階層を合成（地合いで重み可変） ---
    wQ, wS, wR = get_score_weights(mkt_score)

    total_score = quality_score * wQ + setup_score * wS + regime_score * wR

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "score_quality": quality_score,
        "score_setup": setup_score,
        "score_regime": regime_score,
        "score_final": float(total_score),
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "atr": atr,
        "vola20": vola20,
        "hist": hist,
    }


# ============================================================
# IN価格ロジック（3〜10日スイング専用）
# ============================================================
def compute_entry_price(
    close: pd.Series,
    ma5: float,
    ma20: float,
    atr: float,
) -> float:
    """
    “今日から3〜10日スイングで勝ちやすい” IN価格
    - ベースは MA20 付近
    - ATR の 0.5 倍分だけ下方向にずらす（押し目をしっかり待つ）
    - 直近安値を割りすぎないように補正
    - 強トレンド時は少しだけ浅めに
    """
    price = float(close.iloc[-1])
    last_low = float(close.iloc[-5:].min())

    # 基本は MA20
    target = ma20

    # ATR で押し目を深く取りに行く（0.5倍）
    if atr and atr > 0:
        target = target - atr * 0.5

    # 強い上昇トレンド：MA5 > MA20 のときは少し上寄せ（深追いしすぎない）
    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3

    # 現値より上になってしまったら、現値少し下に補正
    if target > price:
        target = price * 0.995

    # 直近安値より下になり過ぎたら、「安値割れはしない」前提で少し上に補正
    if target < last_low:
        target = last_low * 1.02

    return round(float(target), 1)


# ============================================================
# 残りRR（保有ポジション評価用）
# ============================================================
def calc_residual_rr(
    current_price: float,
    entry_price: float,
    tp_pct: float = CORE_TP_PCT,
    sl_pct: float = CORE_SL_PCT,
) -> Optional[float]:
    """
    いまの価格から見た「残りRR」を計算する。
    - TP/SL は entry 基準で固定（+8% / -4%）
    - 今値から TP までの距離 / 今値から SL までの距離
    """
    if not (np.isfinite(current_price) and np.isfinite(entry_price)):
        return None
    if current_price <= 0 or entry_price <= 0:
        return None

    tp_price = entry_price * (1.0 + tp_pct)
    sl_price = entry_price * (1.0 + sl_pct)

    # 上方向の残り幅
    up = max(tp_price / current_price - 1.0, 0.0)
    # 下方向の残りリスク
    down = max(1.0 - sl_price / current_price, 0.0)

    if down <= 0:
        return None

    rr = up / down
    if not np.isfinite(rr):
        return None
    return float(rr)


# ============================================================
# TP / SL ロジック（ボラ＆地合い＆ATR＆直近高値ベース）
#  ※ Coreは +8% / -4% で固定するので、この関数は今後拡張用。
# ============================================================
def calc_candidate_tp_sl(
    vola20: float,
    mkt_score: int,
    atr_ratio: Optional[float],
    swing_upside: Optional[float],
) -> Tuple[float, float]:
    """
    ボラ・地合い・ATR・直近高値までの距離から利確 / 損切りの % を決める
    戻り値: (tp_pct, sl_pct) 例: 0.10, -0.04
    現状 Core は CORE_TP_PCT / CORE_SL_PCT を使う。
    """
    # ここでは安全レンジだけを決めておき、Core では固定値を使用
    tp = CORE_TP_PCT
    sl = CORE_SL_PCT
    return tp, sl


# ============================================================
# SOX / NVDA / 為替・指数を加味した地合い補正
# ============================================================
def enhance_market_score() -> Dict:
    """
    calc_market_score() の結果に
    - SOX / NVDA
    - USDJPY（円高・円安）
    - 日経平均
    の5日騰落を少しだけ上乗せして、
    日本株スイングの実需に寄せる。
    """
    base = calc_market_score()
    # utils.market.calc_market_score が dict か int かを吸収
    if isinstance(base, dict):
        score = float(base.get("score", 50))
        comment = str(base.get("comment", ""))
        info = dict(base)
    else:
        score = float(base)
        comment = ""
        info = {"score": int(score), "comment": comment}

    # 安全側に初期値クリップ
    score = float(np.clip(score, 0.0, 100.0))

    # --- 日経平均の5日騰落 ---
    try:
        nikkei = yf.Ticker("^N225").history(period="6d")
        if nikkei is not None and not nikkei.empty and len(nikkei) >= 2:
            n_chg = float(nikkei["Close"].iloc[-1] / nikkei["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(n_chg / 2.5, -6.0, 6.0))
    except Exception as e:
        print("[WARN] ^N225 fetch failed:", e)

    # --- 半導体指数（SOX） ---
    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(sox_chg / 3.0, -5.0, 5.0))
    except Exception as e:
        print("[WARN] ^SOX fetch failed:", e)

    # --- NVDA 単体 ---
    try:
        nvda = yf.Ticker("NVDA").history(period="6d")
        if nvda is not None and not nvda.empty and len(nvda) >= 2:
            nvda_chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(nvda_chg / 4.0, -4.0, 4.0))
    except Exception as e:
        print("[WARN] NVDA fetch failed:", e)

    # --- 為替（USDJPY） ---
    try:
        fx = yf.Ticker("JPY=X").history(period="6d")
        if fx is not None and not fx.empty and len(fx) >= 2:
            fx_chg = float(fx["Close"].iloc[-1] / fx["Close"].iloc[0] - 1.0) * 100.0
            # 円安方向（USDJPY上昇）は大型輸出に追い風
            score += float(np.clip(fx_chg / 4.0, -3.0, 3.0))
    except Exception as e:
        print("[WARN] FX JPY=X fetch failed:", e)

    score = float(np.clip(round(score), 0, 100))
    info["score"] = int(score)
    if not info.get("comment"):
        # コメントがなければざっくりコメントを付ける
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
# スクリーニング（Top15 → 最終5）
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
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

        # 決算前後 ±N日 は除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue

        # 地合い連動の最低ライン
        if base_score < min_score:
            continue

        info = score_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            hist=hist,
            score_raw=base_score,
            mkt_score=mkt_score,
            sector_strength=sector_strength,
        )
        raw_candidates.append(info)

    # TopN 抽出（スコア最終版でソート）
    raw_candidates.sort(key=lambda x: x["score_final"], reverse=True)
    topN = raw_candidates[:SCREENING_TOP_N]

    # TopN から最終 5 銘柄
    final_list: List[Dict] = []
    for c in topN:
        close = c["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, c["ma5"], c["ma20"], c["atr"])

        price = float(c["price"])

        # Core用 TP/SL は固定 +8% / -4%
        tp_pct = CORE_TP_PCT
        sl_pct = CORE_SL_PCT
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        # 残りRR（IN 時点想定）= 2.0R 固定
        rr = tp_pct / abs(sl_pct) if sl_pct < 0 else 0.0

        price_now = float(c["price"])
        gap_ratio = abs(price_now - entry) / price_now if price_now > 0 else 1.0

        # 今日IN候補か、数日以内IN候補かを分類
        if gap_ratio <= 0.01:
            entry_type = "today"      # 今日から入ってOKゾーン
        else:
            entry_type = "soon"       # 数日以内に押し目を待つゾーン

        final_list.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "sector": c["sector"],
                "score": c["score_final"],
                "price": price_now,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_type": entry_type,
                "rr": rr,
            }
        )

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


# ============================================================
# ポジションRR評価
# ============================================================
def evaluate_positions_rr(pos_df: Optional[pd.DataFrame]) -> List[Dict]:
    """
    positions.csv から現在RRを計算する。
    必須: ticker と entry_price 相当の列。
    """
    results: List[Dict] = []
    if pos_df is None or pos_df.empty:
        return results

    # entry price 推定
    def get_entry(row: pd.Series) -> Optional[float]:
        for col in ["entry_price", "avg_price", "price", "cost"]:
            if col in row.index and pd.notna(row[col]):
                try:
                    v = float(row[col])
                    if v > 0:
                        return v
                except Exception:
                    continue
        return None

    price_cache: Dict[str, float] = {}

    for _, row in pos_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = get_entry(row)
        if entry_price is None:
            continue

        # 現値取得
        if ticker in price_cache:
            current = price_cache[ticker]
        else:
            hist = fetch_history(ticker, period="20d")
            if hist is None or hist.empty:
                continue
            current = float(hist["Close"].iloc[-1])
            price_cache[ticker] = current

        rr_now = calc_residual_rr(current_price=current, entry_price=entry_price)
        if rr_now is None:
            continue

        results.append(
            {
                "ticker": ticker,
                "entry_price": entry_price,
                "current_price": current,
                "rr": rr_now,
            }
        )

    return results


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
    pos_df: Optional[pd.DataFrame],
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i + 1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs[:3])
        ]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    # イベント
    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし（通常）"]

    # スクリーニング（Top → 最終）
    core_list = run_screening(today_date, mkt_score)
    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]

    # ポジションRR評価
    pos_rr_list = evaluate_positions_rr(pos_df)

    # 乗り換え（Chain）推奨判定
    chain_lines: List[str] = []
    if core_list and pos_rr_list:
        # 新しい本命は Core の先頭
        new_core = core_list[0]
        new_rr = float(new_core.get("rr", 0.0))

        for p in pos_rr_list:
            old_rr = float(p["rr"])
            if new_rr > old_rr and new_rr >= 2.0:
                chain_lines.append(
                    f"- {p['ticker']}: 現在RR:{old_rr:.1f}R → 本命 {new_core['ticker']} {new_core['name']} (RR:{new_rr:.1f}R) への乗り換え候補"
                )

    # ロット計算用（1銘柄あたりの許容リスク）
    per_trade_risk = est_asset * RISK_PER_TRADE if est_asset > 0 else 30_000.0

    lines: List[str] = []

    # --- ヘッダー / 結論 ---
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

    # --- セクター ---
    lines.append("◆ 今日のTOPセクター（5日騰落）")
    lines.append(sec_text)
    lines.append("")

    # --- イベント ---
    lines.append("◆ 今日のイベント・警戒")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    # --- Core候補 Aランク（今日IN） ---
    lines.append(f"◆ Core候補 Aランク（今日IN候補 最大{MAX_FINAL_STOCKS}）")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            rr = float(c.get("rr", 2.0))
            # INゾーン（±0.5%）
            center = c["entry"]
            lower = center * (1.0 - IN_ZONE_HALF_PCT)
            upper = center * (1.0 + IN_ZONE_HALF_PCT)

            # ロット（100株単位）
            risk_per_share = max(center - c["sl_price"], 0.0)
            if risk_per_share > 0 and per_trade_risk > 0:
                raw_shares = int(per_trade_risk / risk_per_share)
                shares_100 = (raw_shares // 100) * 100
                if shares_100 < 100:
                    shares_100 = 0
            else:
                shares_100 = 0

            loss_yen = shares_100 * (center - c["sl_price"]) if shares_100 > 0 else 0.0
            gain_yen = shares_100 * (c["tp_price"] - center) if shares_100 > 0 else 0.0
            est_amount = shares_100 * center

            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(
                f"    ・INゾーン: {lower:.1f}〜{upper:.1f}（中心{center:.1f}）"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}） 損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}） RR:{rr:.1f}R"
            )
            if shares_100 > 0:
                lines.append(
                    f"    ・推奨: {shares_100}株 ≒{est_amount:,.0f}円 / 損失~{loss_yen:,.0f}円 利確~{gain_yen:,.0f}円"
                )
            lines.append("")

    # --- Core候補 Aランク（数日以内IN） ---
    lines.append("◆ Core候補 Aランク（数日以内IN候補）")
    if not soon_list:
        lines.append("数日以内IN候補なし")
    else:
        for c in soon_list:
            rr = float(c.get("rr", 2.0))
            center = c["entry"]
            lower = center * (1.0 - IN_ZONE_HALF_PCT)
            upper = center * (1.0 + IN_ZONE_HALF_PCT)

            risk_per_share = max(center - c["sl_price"], 0.0)
            if risk_per_share > 0 and per_trade_risk > 0:
                raw_shares = int(per_trade_risk / risk_per_share)
                shares_100 = (raw_shares // 100) * 100
                if shares_100 < 100:
                    shares_100 = 0
            else:
                shares_100 = 0

            loss_yen = shares_100 * (center - c["sl_price"]) if shares_100 > 0 else 0.0
            gain_yen = shares_100 * (c["tp_price"] - center) if shares_100 > 0 else 0.0
            est_amount = shares_100 * center

            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(
                f"    ・理想IN: {center:.1f} ゾーン:{lower:.1f}〜{upper:.1f}"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}% 損切:{c['sl_pct']*100:.1f}% RR:{rr:.1f}R"
            )
            if shares_100 > 0:
                lines.append(
                    f"    ・推奨: {shares_100}株 ≒{est_amount:,.0f}円 / 損失~{loss_yen:,.0f}円 利確~{gain_yen:,.0f}円"
                )
            lines.append("")

    # --- 本日の建て玉最大金額 ---
    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍 / MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    # --- ポジションサマリ ---
    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip() if pos_text else "ポジション情報なし")
    lines.append("")

    # --- ポジション入れ替え（Chain）候補 ---
    lines.append("◆ ポジション入れ替え候補（RRベース）")
    if not chain_lines:
        lines.append("- 本日明確な入れ替え候補なし（既存ポジも許容範囲）")
    else:
        lines.extend(chain_lines)

    # ここまでがロング版
    long_report = "\n".join(lines)

    # --- ショート（要約）版 ---
    short_lines: List[str] = []
    short_lines.append(f"📅 {today_str} stockbotTOM 要約")
    short_lines.append(f"- 地合い: {mkt_score} / レバ目安: {rec_lev:.1f}倍")
    if core_list:
        best = core_list[0]
        rr = float(best.get("rr", 2.0))
        short_lines.append(
            f"- 本命: {best['ticker']} {best['name']} Score:{best['score']:.1f} [{best['sector']}]"
        )
        short_lines.append(
            f"  IN:{best['entry']:.1f} RR:{rr:.1f}R TP:+{best['tp_pct']*100:.1f}% SL:{best['sl_pct']*100:.1f}%"
        )
    else:
        short_lines.append("- 本命候補なし（今日は無理に攻めない日）")
    short_lines.append(f"- MAX建て玉: 約{max_pos:,}円")

    short_report = "\n".join(short_lines)

    # ロング版とショート版を両方返す（送信時に分割）
    return long_report + "\n\n-----\n\n" + short_report


# ============================================================
# LINE送信（分割対応）
# ============================================================
def send_line(text: str) -> None:
    """
    Cloudflare Worker 経由で LINE へ送信。
    長文は 3900 文字ごとに分割して送る。
    """
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（print のみ）")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信に失敗:", e)
            print(ch)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い（元の calc_market_score に SOX / NVDA / 為替 等を上乗せ）
    mkt = enhance_market_score()

    # ポジション（推定資産 / レバ等）
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
        pos_df=pos_df,
    )

    # ログ出力
    print(report)

    # LINE 送信（自動分割）
    send_line(report)


if __name__ == "__main__":
    main()
