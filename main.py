from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions, compute_positions_rr
from utils import rr
from utils.util import jst_today_str


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3
EARNINGS_EXCLUDE_DAYS = 3
RR_SWAP_DIFF_THRESHOLD = 0.8


def jst_today_date():
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
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


def build_event_warnings(today) -> List[str]:
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

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today) -> bool:
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
        except Exception as e:
            print(f"[WARN] fetch history failed {ticker} (try {attempt+1}): {e}")
    return None


def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
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


def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    strength: Dict[str, float] = {}
    for rank, (name, chg) in enumerate(secs[:5]):
        base = 6 - rank
        boost = max(chg, 0.0) * 0.3
        strength[name] = base + boost
    return strength


def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    hist: pd.DataFrame,
    base_score: float,
    mkt_score: int,
    sector_strength: Dict[str, float],
    rr_info: Dict[str, float],
) -> Dict:
    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])

    setup_score = 0.0
    if ma5 > ma20 > ma60:
        setup_score += 12.0
    elif ma20 > ma5 > ma60:
        setup_score += 6.0
    elif ma20 > ma60 > ma5:
        setup_score += 3.0

    regime_score = 0.0
    regime_score += (mkt_score - 50) * 0.12
    if sector_strength:
        regime_score += sector_strength.get(sector, 0.0)

    wQ = 0.7
    wS = 1.0
    wR = 0.6
    total_score = base_score * wQ + setup_score * wS + regime_score * wR

    tp_pct = float(rr_info.get("tp_pct", 0.0))
    sl_pct = float(rr_info.get("sl_pct", -0.04))
    rr_val = float(rr_info.get("rr", 0.0))

    price_entry = price
    tp_price = price_entry * (1.0 + tp_pct)
    sl_price = price_entry * (1.0 + sl_pct)

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "score": float(total_score),
        "price": price,
        "entry": price_entry,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "rr": rr_val,
    }


def enhance_market_score() -> Dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    try:
        sox = yf.Ticker("^SOX").history(period="5d")
        if sox is not None and not sox.empty:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(sox_chg / 2.0, -5.0, 5.0))
    except Exception as e:
        print("[WARN] SOX fetch failed:", e)

    try:
        nvda = yf.Ticker("NVDA").history(period="5d")
        if nvda is not None and not nvda.empty:
            nvda_chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(nvda_chg / 3.0, -4.0, 4.0))
    except Exception as e:
        print("[WARN] NVDA fetch failed:", e)

    score = float(np.clip(round(score), 0, 100))
    mkt["score"] = int(score)
    return mkt


def run_screening(today, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)
    sector_strength = build_sector_strength_map()

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = rr.score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue

        if base_score < min_score:
            continue

        rr_info = rr.compute_tp_sl_rr(hist, mkt_score)
        rr_val = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr_val) or rr_val < 1.5:
            continue

        info = score_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            hist=hist,
            base_score=base_score,
            mkt_score=mkt_score,
            sector_strength=sector_strength,
            rr_info=rr_info,
        )
        candidates.append(info)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:SCREENING_TOP_N]

    final_list: List[Dict] = []
    for c in top:
        price = c["price"]
        entry = c["entry"]
        gap_ratio = abs(price - entry) / price if price > 0 else 1.0
        entry_type = "today" if gap_ratio <= 0.01 else "soon"
        c["entry_type"] = entry_type
        final_list.append(c)

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


def compute_swap_candidates(
    positions_rr: Dict[str, Dict[str, float]],
    core_list: List[Dict],
    diff_threshold: float = RR_SWAP_DIFF_THRESHOLD,
) -> List[Dict]:
    if not positions_rr:
        return []

    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    if not today_list:
        return []

    best = max(today_list, key=lambda x: x.get("rr", 0.0))
    best_rr = float(best.get("rr", 0.0))
    if not np.isfinite(best_rr):
        return []

    swaps: List[Dict] = []
    for ticker, info in positions_rr.items():
        rr_old = float(info.get("rr", 0.0))
        if not np.isfinite(rr_old):
            continue
        diff = best_rr - rr_old
        if diff >= diff_threshold:
            swaps.append(
                {
                    "from_ticker": ticker,
                    "to_ticker": best["ticker"],
                    "from_rr": rr_old,
                    "to_rr": best_rr,
                    "diff": diff,
                }
            )

    return swaps


def build_report(
    today_str: str,
    today_date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
    core_list: List[Dict],
    swap_list: List[Dict],
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i + 1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs[:10])
        ]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし（通常モード）"]

    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]

    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 推定運用資産ベース: 約{est_asset_int:,}円")
    lines.append("")
    lines.append("◆ 今日のTOPセクター（5日騰落率・上位10）")
    lines.append(sec_text)
    lines.append("")
    lines.append("◆ 今日のイベント・警戒情報")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    lines.append(f"◆ Core候補 Aランク（本命押し目・今日IN候補・最大{MAX_FINAL_STOCKS}銘柄）")
    if not today_list:
        lines.append("今日すぐにINできる本命Aランク候補はなし。")
    else:
        for c in today_list:
            lines.append(
                f"- {c['ticker']} {c['name']}  Score:{c['score']:.1f} RR:{c['rr']:.2f}R 現値:{c['price']:.1f}"
            )
            lines.append(f"    ・IN目安: {c['entry']:.1f}")
            lines.append(
                f"    ・利確目安: +{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}）"
            )
            lines.append(
                f"    ・損切り目安: {c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）"
            )
            lines.append("")

    lines.append("◆ Core候補 Aランク（数日以内の押し目待ち候補）")
    if not soon_list:
        lines.append("数日以内の押し目待ちAランク候補なし。")
    else:
        for c in soon_list:
            lines.append(
                f"- {c['ticker']} {c['name']}  Score:{c['score']:.1f} RR:{c['rr']:.2f}R 現値:{c['price']:.1f}"
            )
            lines.append(f"    ・理想IN目安: {c['entry']:.1f}")
            lines.append(
                f"    ・利確目安: +{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}）"
            )
            lines.append(
                f"    ・損切り目安: {c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）"
            )
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円")
    lines.append("")

    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())
    lines.append("")

    lines.append(f"◆ スワップ候補（RR差 >= +{RR_SWAP_DIFF_THRESHOLD:.1f}R）")
    if not swap_list:
        lines.append("現在、明確なスワップ候補なし。")
    else:
        for s in swap_list:
            lines.append(
                f"- {s['from_ticker']} → {s['to_ticker']} "
                f"RR差:+{s['diff']:.2f}R（保有{s['from_rr']:.2f}R → 候補{s['to_rr']:.2f}R）"
            )

    return "\n".join(lines)


def send_line(text: str) -> None:
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


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    core_list = run_screening(today_date, mkt_score)

    positions_rr = compute_positions_rr(pos_df, mkt_score)
    swap_list = compute_swap_candidates(positions_rr, core_list)

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
        core_list=core_list,
        swap_list=swap_list,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()