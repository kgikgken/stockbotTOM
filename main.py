from __future__ import annotations

import os
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d, sector_rank_map
from utils.rr import classify_setup, build_trade_plan, TradePlan
from utils.scoring import passes_universe_filters, estimate_pwin, compute_ev, regime_multiplier
from utils.position import load_positions, analyze_positions

# ============================================================
# 設定（Swing 1〜7日）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

WORKER_URL = os.getenv("WORKER_URL")

# 決算フィルタ: ±N日（暦日近似）
EARNINGS_EXCLUDE_DAYS = 3

# Universe
PRICE_MIN = 200.0
PRICE_MAX = 15000.0
ADV_MIN_JPY = 100_000_000.0
ATR_PCT_MIN = 0.015

# Sector
SECTOR_TOP_N = 5  # 原則上位5

# 足切り
R_MIN = 2.2
EV_MIN = 0.40
EXPECTED_DAYS_MAX = 5.0
R_PER_DAY_MIN = 0.50

# 分散
MAX_FINAL = 5
MAX_PER_SECTOR = 2
MAX_CORR = 0.75

# NO-TRADE day（最終確定）
NO_TRADE_AVG_ADJEV_MIN = 0.30
NO_TRADE_GU_RATIO = 0.60

# 監視表示
WATCH_MAX = 10


# ============================================================
# yfinance
# ============================================================
def fetch_history(ticker: str, period: str = "320d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


# ============================================================
# events
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        label = str(r.get("label", "")).strip()
        if not label:
            continue
        out.append(
            dict(
                label=label,
                kind=str(r.get("kind", "")).strip(),
                date=str(r.get("date", "")).strip(),
                time=str(r.get("time", "")).strip(),
                datetime=str(r.get("datetime", "")).strip(),
            )
        )
    return out


def build_event_warnings(today_date) -> Tuple[List[str], float]:
    """
    戻り値:
      - warns: 表示用
      - event_penalty: 0.75 or 1.0（直近イベントがあれば減衰）
    """
    events = load_events()
    warns: List[str] = []
    penalty = 1.0

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

        if 0 <= delta <= 1:
            penalty = min(penalty, 0.75)

    if not warns:
        warns.append("- 特になし")

    return warns, float(penalty)


# ============================================================
# earnings filter
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    keep = []
    for x in d:
        if x is None or pd.isna(x):
            keep.append(True)
            continue
        try:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return df[keep]


# ============================================================
# correlation
# ============================================================
def corr_20d(hist_a: pd.DataFrame, hist_b: pd.DataFrame) -> float:
    try:
        a = hist_a["Close"].astype(float).pct_change(fill_method=None).tail(21)
        b = hist_b["Close"].astype(float).pct_change(fill_method=None).tail(21)
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 10:
            return 0.0
        return float(df.corr().iloc[0, 1])
    except Exception:
        return 0.0


# ============================================================
# screening core
# ============================================================
def screen_swing(today_date, mkt: Dict[str, object]) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    """
    戻り値:
      - finals: 本命（最大5）
      - watch: 監視（最大10）
      - meta: ヘッダ用
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], [], {"reason": "universe読み込み失敗"}

    t_col = "ticker" if "ticker" in uni.columns else ("code" if "code" in uni.columns else None)
    if not t_col:
        return [], [], {"reason": "ticker列なし"}

    # 決算除外（新規）
    uni = filter_earnings(uni, today_date)

    # セクター順位
    sector_ranks = sector_rank_map(top_n=SECTOR_TOP_N)
    top_sectors = set(sector_ranks.keys())

    market_score = int(mkt.get("score", 50))
    d_market_3d = int(mkt.get("d_market_3d", 0))

    # イベント
    event_warns, event_penalty = build_event_warnings(today_date)

    # 地合いNO-TRADE（前段）
    no_trade_by_mkt = (market_score < 45) or (d_market_3d <= -5 and market_score < 55)

    cands: List[Dict] = []
    watch: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))
        sector_rank = sector_ranks.get(sector)

        # 原則: 上位セクターのみ
        if sector_rank is None:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        ok, uni_meta = passes_universe_filters(
            hist,
            price_min=PRICE_MIN,
            price_max=PRICE_MAX,
            adv_min=ADV_MIN_JPY,
            atrp_min=ATR_PCT_MIN,
        )
        if not ok:
            continue

        setup = classify_setup(hist)
        if setup is None:
            continue

        plan = build_trade_plan(hist, setup=setup)
        if plan is None:
            continue

        # RR
        if plan.r < R_MIN:
            continue

        # 監視ルール（追いかけ禁止の出力枠）
        if plan.gu_flag or plan.in_distance_atr > 0.8:
            watch.append(dict(ticker=ticker, name=name, sector=sector, sector_rank=int(sector_rank), setup=setup, plan=plan))
            continue

        # 速度
        if plan.expected_days > EXPECTED_DAYS_MAX or plan.r_per_day < R_PER_DAY_MIN:
            watch.append(dict(ticker=ticker, name=name, sector=sector, sector_rank=int(sector_rank), setup=setup, plan=plan))
            continue

        # Pwin/EV/AdjEV
        sec_rank01 = 1.0 - (sector_rank - 1) / max(1, SECTOR_TOP_N - 1)
        pwin = estimate_pwin(hist, plan, sector_rank01=sec_rank01, adv20=uni_meta["adv20"], market_score=market_score)
        ev = compute_ev(pwin, plan.r)
        adjev = float(ev * regime_multiplier(market_score, d_market_3d, event_penalty))

        if ev < EV_MIN:
            continue

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                sector_rank=int(sector_rank),
                setup=str(setup),
                plan=plan,
                pwin=float(pwin),
                ev=float(ev),
                adjev=float(adjev),
                adv20=float(uni_meta["adv20"]),
                hist=hist,
            )
        )

    # 追加のNO-TRADE（候補の質）
    gu_ratio = float(np.mean([1.0 if c["plan"].gu_flag else 0.0 for c in cands])) if cands else 0.0
    avg_adjev = float(np.mean([c["adjev"] for c in cands])) if cands else 0.0

    no_trade_reason = ""
    if no_trade_by_mkt:
        no_trade_reason = "地合い条件"
    elif cands and avg_adjev < NO_TRADE_AVG_ADJEV_MIN:
        no_trade_reason = f"平均AdjEV<{NO_TRADE_AVG_ADJEV_MIN:.2f}"
    elif cands and gu_ratio >= NO_TRADE_GU_RATIO:
        no_trade_reason = "GU銘柄が多い"

    # 本命候補の並び
    cands.sort(key=lambda x: (x["adjev"], x["plan"].r_per_day, x["plan"].r), reverse=True)

    # 分散（セクター/相関）
    finals: List[Dict] = []
    sector_counts: Dict[str, int] = {}

    for c in cands:
        if len(finals) >= MAX_FINAL:
            break

        sec = c["sector"]
        if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
            continue

        ok_corr = True
        for f in finals:
            co = corr_20d(c["hist"], f["hist"])
            if np.isfinite(co) and co > MAX_CORR:
                ok_corr = False
                break
        if not ok_corr:
            continue

        finals.append(c)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # 監視リスト整形（上位だけ）
    watch_sorted = sorted(
        watch,
        key=lambda x: (x["plan"].r_per_day, x["plan"].r),
        reverse=True,
    )[:WATCH_MAX]

    meta = dict(
        trade_ok=(no_trade_reason == ""),
        no_trade_reason=no_trade_reason,
        market_score=market_score,
        d_market_3d=d_market_3d,
        event_warns=event_warns,
        event_penalty=event_penalty,
        gu_ratio=gu_ratio,
        avg_adjev=avg_adjev,
        top_sectors=list(top_sectors),
    )

    if no_trade_reason:
        return [], watch_sorted, meta

    return finals, watch_sorted, meta


# ============================================================
# report
# ============================================================
def _lev_from_market(market_score: int, d_market_3d: int) -> float:
    if market_score >= 60 and d_market_3d >= 0:
        return 2.0
    if market_score >= 45:
        return 1.7
    return 0.0


def build_report(today_str: str, today_date, mkt: Dict[str, object], pos_text: str, total_asset: float) -> str:
    finals, watch, meta = screen_swing(today_date, mkt)

    lev = _lev_from_market(int(meta["market_score"]), int(meta["d_market_3d"]))
    max_pos = int(total_asset * lev) if lev > 0 else 0

    sectors = top_sectors_5d(SECTOR_TOP_N)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    if meta["trade_ok"]:
        lines.append("✅ 新規可（条件クリア）")
    else:
        lines.append("🚫 本日は新規見送り（条件該当）")
        lines.append(f"- 理由: {meta['no_trade_reason']}")

    lines.append(f"- 地合い: {meta['market_score']}点 ({mkt.get('comment', '')})")
    lines.append(f"- ΔMarketScore_3d: {meta['d_market_3d']:+d}")
    if lev > 0:
        lev_comment = "中立（厳選・押し目中心）" if meta["market_score"] < 60 else "攻め"
        lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
        lines.append(f"- MAX建玉: 約{max_pos:,}円\n")
    else:
        lines.append("- レバ: 0.0倍")
        lines.append("- MAX建玉: 0円\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors, 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(meta["event_warns"])
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if finals:
        avg_rr = float(np.mean([c["plan"].r for c in finals]))
        avg_ev = float(np.mean([c["ev"] for c in finals]))
        avg_adjev = float(np.mean([c["adjev"] for c in finals]))
        avg_rpd = float(np.mean([c["plan"].r_per_day for c in finals]))
        lines.append(f"  候補数:{len(finals)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_adjev:.2f} / 平均R/day:{avg_rpd:.2f}\n")

        # 本命：上位1-2（AdjEVで）
        core = finals[:2]
        rest = finals[2:]

        lines.append("🎯 本命（1〜2銘柄）")
        for c in core:
            lines.extend(_format_candidate(c, star=True))
        lines.append("")

        if rest:
            lines.append("👀 監視・指値")
            for c in rest:
                lines.extend(_format_candidate(c, star=False))
            lines.append("")
    else:
        lines.append("- 該当なし\n")

    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in watch[:WATCH_MAX]:
            lines.extend(_format_watch(w))
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


def _format_candidate(c: Dict, star: bool) -> List[str]:
    plan: TradePlan = c["plan"]
    close_now = float(c["hist"]["Close"].astype(float).iloc[-1])
    gap_pct = (close_now / plan.in_center - 1.0) * 100.0 if plan.in_center > 0 else 0.0
    setup_j = "A(押し目)" if c["setup"] == "A" else "B(ブレイク)"
    star_mark = " ⭐" if star else ""
    gu = "Y" if plan.gu_flag else "N"
    return [
        f"- {c['ticker']} {c['name']} [{c['sector']}]{star_mark}",
        f"  形:{setup_j}  RR:{plan.r:.2f}  AdjEV:{c['adjev']:.2f}  R/day:{plan.r_per_day:.2f}",
        f"  IN:{plan.in_center:.1f}（帯:{plan.in_low:.1f}〜{plan.in_high:.1f}） 現在:{close_now:.1f} ({gap_pct:+.2f}%)  ATR:{plan.atr:.1f}  GU:{gu}",
        f"  STOP:{plan.stop:.1f}  TP1:{plan.tp1:.1f}  TP2:{plan.tp2:.1f}  ExpectedDays:{plan.expected_days:.1f}  行動:{plan.action}",
        "",
    ]


def _format_watch(w: Dict) -> List[str]:
    plan: TradePlan = w["plan"]
    setup_j = "A(押し目)" if w["setup"] == "A" else "B(ブレイク)"
    gu = "Y" if plan.gu_flag else "N"
    reason = "GU/追いかけ" if (plan.gu_flag or plan.in_distance_atr > 0.8) else "速度/効率"
    return [
        f"- {w['ticker']} {w['name']} [{w['sector']}]",
        f"  形:{setup_j}  RR:{plan.r:.2f}  R/day:{plan.r_per_day:.2f}  理由:{reason}  行動:{plan.action}  GU:{gu}",
    ]


# ============================================================
# LINE
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    for ch in [text[i:i + 3800] for i in range(0, len(text), 3800)]:
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE]", r.status_code)


# ============================================================
# main
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
