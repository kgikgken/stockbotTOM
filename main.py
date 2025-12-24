from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score, market_momentum_3d
from utils.sector import top_sectors_5d
from utils.scoring import (
    trend_gate,
    detect_setup_type,
    calc_in_zone,
    compute_universe_filters,
    compute_candidate_metrics,
    build_portfolio_selection,
    decide_action,
)
from utils.position import load_positions, analyze_positions


# ============================================================
# Swing Screener v1.4 / ログなし最終寄せ
# LINE送信仕様は「届いたやつ」互換：POST json={"text": "..."} / 3800分割
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SECTOR_TOP_N = 5

# 最終出力
MAX_FINAL = 5
WATCH_MAX = 10

# 決算前後
EARNINGS_EXCLUDE_DAYS = 3

# Universe
PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 100_000_000
ATR_PCT_MIN = 0.015

# Market / No-trade
NO_TRADE_SCORE = 45
NO_TRADE_DELTA3 = -5
NO_TRADE_DELTA3_SCORECAP = 55


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


def load_events(path: str = EVENTS_PATH) -> List[str]:
    if not os.path.exists(path):
        return ["- 特になし"]

    try:
        df = pd.read_csv(path)
    except Exception:
        return ["- 特になし"]

    today = jst_today_date()
    out: List[str] = []

    for _, r in df.iterrows():
        dt = parse_event_datetime_jst(
            str(r.get("datetime", "")),
            str(r.get("date", "")),
            str(r.get("time", "")),
        )
        if dt is None:
            continue

        delta = (dt.date() - today).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            label = str(r.get("label", "イベント"))
            out.append(f"⚠ {label}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

    return out if out else ["- 特になし"]


def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        ed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    keep = []
    for d in ed:
        if pd.isna(d):
            keep.append(True)
        else:
            keep.append(abs((d - today_date).days) > EARNINGS_EXCLUDE_DAYS)
    return df[keep]


def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return

    chunk = 3800
    for i in range(0, len(text), chunk):
        ch = text[i : i + chunk]
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE]", r.status_code, str(r.text)[:120])


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    d3 = market_momentum_3d()

    sector_rank = top_sectors_5d(top_n=SECTOR_TOP_N)
    sector_top_names = [s for s, _ in sector_rank]

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))

    # Universe
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        uni = pd.DataFrame()

    if not uni.empty:
        tcol = "ticker" if "ticker" in uni.columns else ("code" if "code" in uni.columns else None)
    else:
        tcol = None

    candidates: List[Dict] = []

    if tcol is not None:
        uni = filter_earnings(uni, today_date)

        events_list = load_events()
        events_soon = any(s.startswith("⚠") for s in events_list)

        for _, row in uni.iterrows():
            ticker = str(row.get(tcol, "")).strip()
            if not ticker:
                continue

            hist = fetch_history(ticker)
            if hist is None or len(hist) < 120:
                continue

            sector = str(row.get("sector", row.get("industry_big", "不明")))
            name = str(row.get("name", ticker))

            ok, uni_info = compute_universe_filters(
                hist,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                adv20_min=ADV20_MIN,
                atr_pct_min=ATR_PCT_MIN,
            )
            if not ok:
                continue

            if sector not in sector_top_names:
                continue
            sector_rank_i = sector_top_names.index(sector) + 1

            if not trend_gate(hist):
                continue

            setup = detect_setup_type(hist)
            if setup not in ("A", "B"):
                continue

            in_low, in_high, in_center, atr = calc_in_zone(hist, setup)
            action = decide_action(hist, in_center, in_low, in_high, atr)

            cm = compute_candidate_metrics(
                hist=hist,
                setup=setup,
                sector_rank=sector_rank_i,
                mkt_score=int(mkt.get("score", 50)),
                d_mkt_3d=int(d3),
                events_soon=bool(events_soon),
            )
            if cm is None:
                continue

            candidates.append(
                dict(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    sector_rank=sector_rank_i,
                    setup=setup,
                    price_now=float(hist["Close"].iloc[-1]),
                    atr=float(atr),
                    in_low=float(in_low),
                    in_high=float(in_high),
                    in_center=float(in_center),
                    action=action,
                    **uni_info,
                    **cm,
                )
            )

    # NO-TRADE判定（候補平均AdjEV / GU比率）
    mkt_score = int(mkt.get("score", 50))
    no_trade_reasons: List[str] = []

    if mkt_score < NO_TRADE_SCORE:
        no_trade_reasons.append(f"MarketScore<{NO_TRADE_SCORE}")
    if d3 <= NO_TRADE_DELTA3 and mkt_score < NO_TRADE_DELTA3_SCORECAP:
        no_trade_reasons.append(f"Δ3d<={NO_TRADE_DELTA3} & MarketScore<{NO_TRADE_DELTA3_SCORECAP}")

    if candidates:
        avg_adjev = float(np.mean([c["adj_ev"] for c in candidates]))
        if avg_adjev < 0.30:
            no_trade_reasons.append("平均AdjEV<0.30")
        gu_ratio = float(np.mean([1.0 if c.get("gu_flag") else 0.0 for c in candidates]))
        if gu_ratio >= 0.60:
            no_trade_reasons.append("GU比率>=60%")
    else:
        avg_adjev = 0.0

    trade_ok = (len(no_trade_reasons) == 0)

    main_list, watch_list = build_portfolio_selection(
        candidates=candidates,
        max_final=MAX_FINAL,
        watch_max=WATCH_MAX,
        max_per_sector=2,
        corr_lookback=20,
        corr_max=0.75,
    )

    # レポート（日本語）
    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    if trade_ok:
        lines.append("✅ 新規可（条件クリア）")
    else:
        lines.append("🚫 新規見送り（条件該当）")
        lines.append("  - " + " / ".join(no_trade_reasons))

    lines.append(f"- 地合い: {mkt_score}点 ({mkt.get('comment','中立')})")
    lines.append(f"- ΔMarketScore_3d: {int(d3):+d}")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sector_rank:
        for i, (s, p) in enumerate(sector_rank, 1):
            lines.append(f"{i}. {s} ({p:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(load_events())
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if trade_ok and main_list:
        avg_rr = float(np.mean([c["rr"] for c in main_list]))
        avg_ev = float(np.mean([c["ev"] for c in main_list]))
        avg_ad = float(np.mean([c["adj_ev"] for c in main_list]))
        avg_rpd = float(np.mean([c["r_per_day"] for c in main_list]))
        lines.append(f"  候補数:{len(main_list)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_ad:.2f} / 平均R/day:{avg_rpd:.2f}")
        lines.append("")

        lines.append("🎯 本命（最大5）")
        for c in main_list:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] ⭐")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_per_day']:.2f}")
            lines.append(
                f"  IN:{c['in_center']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{'Y' if c.get('gu_flag') else 'N'}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  行動:{c['action']}\n"
            )
    else:
        lines.append("- 該当なし\n")

    if watch_list:
        lines.append("🧠 監視リスト（今日は入らない）")
        for c in watch_list:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] ")
            lines.append(
                f"  形:{c['setup']}  RR:{c['rr']:.2f}  R/day:{c['r_per_day']:.2f}  理由:{c['watch_reason']}  行動:{c['action']}  GU:{'Y' if c.get('gu_flag') else 'N'}"
            )
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    report = "\n".join(lines).strip()
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
