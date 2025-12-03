from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# 既存utilsはすべて利用（機能削減なし）
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
WORKER_URL = os.getenv("WORKER_URL")

# スクリーニング関連
SCREENING_TOP_N = 10        # まずは Top10 まで抽出
MAX_FINAL_STOCKS = 3        # 最終的に LINE に出すのは最大3銘柄

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付 / イベント関連
# ============================================================
def jst_today_date() -> datetime.date:
    """JST の「今日」の date を返す"""
    return datetime.now(timezone(timedelta(hours=9))).date()


# 重要イベント（必要に応じて手動追加していく）
# 例:
# EVENT_CALENDAR = [
#     {"date": "2025-12-10", "label": "米CPI", "kind": "macro"},
#     {"date": "2025-12-13", "label": "FOMC", "kind": "macro"},
#     {"date": "2025-12-18", "label": "NVDA 決算", "kind": "mega-tech"},
# ]
EVENT_CALENDAR: List[Dict[str, str]] = []


def build_event_warnings(today: datetime.date) -> List[str]:
    """イベント接近時の警告メッセージ"""
    warns: List[str] = []
    for ev in EVENT_CALENDAR:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today).days
        # イベントの2日前〜翌日は警告
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
    """株価履歴取得（失敗時 None）"""
    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(f"[WARN] fetch history failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None
    return df


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
    # FutureWarning 対応で fill_method=None
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
    （今までの挙動を維持しつつ、“勝ちやすさ優先” に微調整済）
    """
    if mkt_score >= 70:
        return 1.8, "強め（押し目＋一部ブレイク可）"
    if mkt_score >= 60:
        return 1.5, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "標準（押し目のみ）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規最小ロット）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# Top10用の強化スコアリング（3〜10日スイング向け）
# ============================================================
def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    hist: pd.DataFrame,
    score_raw: float,
    mkt_score: int,
) -> Dict:
    """
    Top10銘柄の内部スコアリング（強化版）
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

    score = 0.0

    # 1. 元スコア（ベース）
    score += float(score_raw) * 1.0

    # 2. トレンド方向（MAの並び）
    trend_score = 0.0
    if ma5 > ma20 > ma60:
        trend_score += 12.0
    elif ma20 > ma5 > ma60:
        trend_score += 6.0
    elif ma20 > ma60 > ma5:
        trend_score += 3.0
    score += trend_score

    # 3. RSI（過熱 / 売られ過ぎの調整）
    if 40 <= rsi <= 65:
        score += 10.0
    elif 30 <= rsi < 40 or 65 < rsi <= 70:
        score += 3.0
    else:
        score -= 6.0

    # 4. ボラティリティの安定感
    if vola20 < 0.02:
        score += 5.0
    elif vola20 > 0.05:
        score -= 4.0

    # 5. ATR（値幅の取りやすさ）
    if atr and price > 0:
        atr_ratio = atr / price
        if 0.015 <= atr_ratio <= 0.035:
            score += 6.0
        elif atr_ratio < 0.01 or atr_ratio > 0.06:
            score -= 5.0

    # 6. 地合いの追い風
    score += (mkt_score - 50) * 0.12

    # 7. 「行き過ぎ高値」を減点（RSI + 5日騰落 + MA乖離）
    if len(close) >= 5:
        ret_5d = close.iloc[-1] / close.iloc[-5] - 1.0
    else:
        ret_5d = 0.0

    ma_spread = 0.0
    if ma20 != 0:
        ma_spread = (ma5 - ma20) / abs(ma20)

    overheat_penalty = 0.0
    if rsi > 72:
        overheat_penalty -= 6.0
    if ret_5d > 0.12:
        overheat_penalty -= 4.0
    if ma_spread > 0.05:
        overheat_penalty -= 4.0

    score += overheat_penalty

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "score_raw": float(score_raw),
        "score_final": float(score),
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
    - ATR で少し下に寄せる
    - 直近安値より下になり過ぎたら補正
    - 強い上昇トレンドではやや高め
    """
    price = float(close.iloc[-1])
    last_low = float(close.iloc[-5:].min())

    # 基本は MA20
    target = ma20

    # ATR で少しだけ下に寄せる
    if atr and atr > 0:
        target = target - atr * 0.3

    # 強い上昇トレンド：MA5 > MA20 のときは少し上寄せ
    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3

    # 現値より上になってしまったら、現値の少し下に補正
    if target > price:
        target = price * 0.995

    # 直近安値より下になり過ぎたら、「安値割れしない」前提で少し上に補正
    if target < last_low:
        target = last_low * 1.02

    return round(float(target), 1)


# ============================================================
# TP / SL ロジック（ボラ＆地合いベース）
# ============================================================
def calc_candidate_tp_sl(vola20: float, mkt_score: int) -> Tuple[float, float]:
    """
    ボラと地合いから利確 / 損切りの % を決める
    戻り値: (tp_pct, sl_pct) 例: 0.10, -0.04
    """
    v = abs(vola20) if np.isfinite(vola20) else 0.03

    if v < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        tp = 0.08
        sl = -0.04
    else:
        tp = 0.12
        sl = -0.06

    # 地合いで微調整
    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl


# ============================================================
# IN価格が「明日〜数日で現実的か？」チェック
# ============================================================
def is_entry_reachable(
    price: float,
    entry: float,
    atr: float,
) -> bool:
    """
    今日〜2営業日くらいで “現実的にタッチし得る” IN かどうかの判定
    - 価格差が 2ATR を大きく超える → NG
    - 7%以上下に置く → 深すぎる押し目として NG
    """
    if not (np.isfinite(price) and np.isfinite(entry)):
        return False

    if entry >= price * 1.01:
        # そもそも現値より上にあるような IN は除外
        return False

    dd = price - entry
    if dd <= 0:
        # ほぼ同値〜少し下はOK
        return True

    # ATR が取れていない場合はざっくり 7% まで許容
    if not atr or atr <= 0:
        return dd <= price * 0.07

    # 通常は 2ATR + 3% くらいを “ギリ届く” とみなす
    max_dd = max(2.0 * atr, price * 0.03)
    if dd > max_dd and dd > price * 0.07:
        return False

    return True


# ============================================================
# スクリーニング（Top10 → 最終3 + セクター分散）
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

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

        # あまりに低スコアな銘柄はそもそも候補にしない
        if base_score < 75:
            continue

        info = score_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            hist=hist,
            score_raw=base_score,
            mkt_score=mkt_score,
        )
        raw_candidates.append(info)

    # Top10 抽出（内部強化スコア順）
    raw_candidates.sort(key=lambda x: x["score_final"], reverse=True)
    top10 = raw_candidates[:SCREENING_TOP_N]

    # Top10 から IN価格・TP/SL を計算しつつフィルタ
    enriched: List[Dict] = []
    for c in top10:
        close = c["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, c["ma5"], c["ma20"], c["atr"])
        tp_pct, sl_pct = calc_candidate_tp_sl(c["vola20"], mkt_score)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        price = c["price"]

        # 「明日〜数日で届き得るIN」だけ残す
        if not is_entry_reachable(price, entry, c["atr"]):
            continue

        enriched.append(
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
            }
        )

    # 強化スコア順に並び替え
    enriched.sort(key=lambda x: x["score"], reverse=True)

    # セクター分散を意識した最終3銘柄選定
    selected: List[Dict] = []
    used_sectors: set[str] = set()

    # 第1パス：なるべくセクターを分散しながらピック
    for c in enriched:
        sec = c["sector"]
        if sec not in used_sectors or len(selected) < 2:
            selected.append(c)
            used_sectors.add(sec)
        if len(selected) >= MAX_FINAL_STOCKS:
            break

    if len(selected) < MAX_FINAL_STOCKS:
        # 第2パス：残り枠はセクター被りOKで上から詰める
        for c in enriched:
            if c in selected:
                continue
            selected.append(c)
            if len(selected) >= MAX_FINAL_STOCKS:
                break

    return selected[:MAX_FINAL_STOCKS]


# ============================================================
# レポート構築
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

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i + 1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs)
        ]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    # イベント警告
    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし（通常モード）"]

    # スクリーニング（Top10 → 最終3 + セクター分散）
    core_list = run_screening(today_date, mkt_score)

    lines: List[str] = []

    # --- ヘッダー / 結論 ---
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 推定運用資産ベース: 約{est_asset_int:,}円")
    lines.append("")

    # --- セクター ---
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text)
    lines.append("")

    # --- イベント ---
    lines.append("◆ 今日のイベント・警戒情報")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    # --- Core候補 Aランク ---
    lines.append(f"◆ Core候補 Aランク（本命押し目・最大{MAX_FINAL_STOCKS}銘柄）")
    if not core_list:
        lines.append("本命Aランク候補なし（今日は無理IN禁止寄り）。")
    else:
        for c in core_list:
            lines.append(
                f"- {c['ticker']} {c['name']}  Score:{c['score']:.1f} 現値:{c['price']:.1f}"
            )
            lines.append(f"    ・IN目安: {c['entry']:.1f}")
            lines.append(
                f"    ・利確目安: +{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}）"
            )
            lines.append(
                f"    ・損切り目安: {c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）"
            )
            lines.append("")

    # --- 建て玉最大金額 ---
    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円")
    lines.append("")

    # --- ポジション分析 ---
    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())

    return "\n".join(lines)


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

    # 地合い
    mkt = calc_market_score()

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
    )

    # ログ出力
    print(report)

    # LINE 送信（自動分割）
    send_line(report)


if __name__ == "__main__":
    main()