from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.day import score_daytrade_candidate


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外
EARNINGS_EXCLUDE_DAYS = 3

# Swing（抽出上限：ここから制約で削る）
SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.30  # vFinal思想に合わせる（EV>=0.3R）

# Day
DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.5

# 表示
SECTOR_TOP_N = 5

# Conditional Aggression（②確定）
LEV_MAX = 2.5


# ============================================================
# 便利
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


# ============================================================
# events.csv
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()

        if not label:
            continue
        events.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str})
    return events


def build_event_warnings(today_date) -> List[str]:
    warns: List[str] = []
    for ev in load_events():
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    return warns or ["- 特になし"]


# ============================================================
# 決算フィルタ
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    try:
        parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    df = df.copy()
    df["earnings_date_parsed"] = parsed

    keep = []
    for d in df["earnings_date_parsed"]:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            delta = abs((d - today_date).days)
            keep.append(delta > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return df[keep]


# ============================================================
# EV(R)
# ============================================================
def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0

    # 勝率仮定（ログ導入後に更新前提）
    if in_rank == "強IN":
        win = 0.45
    elif in_rank == "通常IN":
        win = 0.40
    elif in_rank == "弱めIN":
        win = 0.33
    else:
        win = 0.25

    lose = 1.0 - win
    return float(win * rr - lose * 1.0)


# ============================================================
# Conditional Aggression（②）
# ============================================================
def calc_aggression_level(in_rank: str, rr: float, ev_r: float) -> int:
    """
    AL3: 強IN & RR>=2.8 & EV>=0.6
    AL2: (強IN or 通常IN) & RR>=2.2 & EV>=0.4
    AL1: 通常IN & RR>=1.8 & EV>=0.3
    else: 0（取らない）
    """
    if in_rank == "強IN" and rr >= 2.8 and ev_r >= 0.6:
        return 3
    if in_rank in ("強IN", "通常IN") and rr >= 2.2 and ev_r >= 0.4:
        return 2
    if in_rank == "通常IN" and rr >= 1.8 and ev_r >= 0.3:
        return 1
    return 0


def leverage_from_al(al: int) -> float:
    if al >= 3:
        return 2.3  # ベース（地合いで微調整）
    if al == 2:
        return 1.7
    if al == 1:
        return 1.3
    return 1.0


def adjust_leverage_by_market(base_lev: float, mkt_score: int) -> float:
    lev = float(base_lev)
    # 地合いは「減速要因」：押し目が良ければAL2まで許容、ただしレバ微減
    if mkt_score < 40:
        lev -= 0.35
    elif mkt_score < 45:
        lev -= 0.20
    elif mkt_score >= 70:
        lev += 0.10

    lev = float(np.clip(lev, 1.0, LEV_MAX))
    return lev


# ============================================================
# Swingスクリーニング（候補は多め→制約で絞る）
# ============================================================
def run_swing_candidates(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return []

    uni = filter_earnings(uni, today_date)

    MIN_SCORE = float(SWING_SCORE_MIN)
    RR_MIN = float(SWING_RR_MIN)
    EV_MIN = float(SWING_EV_R_MIN)

    # 地合いで“閾値だけ”微調整（候補数は減らさない）
    if mkt_score >= 70:
        MIN_SCORE -= 3.0
        RR_MIN -= 0.1
    elif mkt_score <= 45:
        MIN_SCORE += 3.0
        RR_MIN += 0.1

    cands: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score) or base_score < MIN_SCORE:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue
        if mkt_score <= 45 and in_rank == "弱めIN":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info["rr"])
        if rr < RR_MIN:
            continue

        ev_r = expected_r_from_in_rank(in_rank, rr)
        if ev_r < EV_MIN:
            continue

        al = calc_aggression_level(in_rank, rr, ev_r)
        if al <= 0:
            continue  # ②ではAL0は「取らない」

        base_lev = leverage_from_al(al)
        rec_lev = adjust_leverage_by_market(base_lev, mkt_score)

        entry = float(rr_info["entry"])
        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        cands.append(dict(
            ticker=ticker,
            name=name,
            sector=sector,
            score=float(base_score),
            rr=float(rr),
            ev_r=float(ev_r),
            in_rank=in_rank,
            al=int(al),
            rec_lev=float(rec_lev),
            entry=float(entry),
            entry_basis=str(rr_info.get("entry_basis", "pullback")),
            price_now=float(price_now) if np.isfinite(price_now) else np.nan,
            gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
            tp_pct=float(rr_info["tp_pct"]),
            sl_pct=float(rr_info["sl_pct"]),
            tp_price=float(rr_info["tp_price"]),
            sl_price=float(rr_info["sl_price"]),
        ))

    # 優先：AL → Score → EV → RR
    cands.sort(key=lambda x: (x["al"], x["score"], x["ev_r"], x["rr"]), reverse=True)
    return cands


def apply_swing_constraints(swing_cands: List[Dict], day_cands: List[Dict]) -> List[Dict]:
    """
    ②仕様の制約:
    - Swing最大3
    - AL3は最大1銘柄
    - 同一銘柄：DayとSwing併用禁止（ここではSwing優先でDay側を落とす想定）
    - 同一セクター：AL3がある場合、他はAL1まで
    """
    if not swing_cands:
        return []

    day_tickers = {c["ticker"] for c in (day_cands or [])}
    # 同一銘柄はSwing側に残す（Day側はreport段で除外）
    swing_cands = [c for c in swing_cands if c["ticker"] not in day_tickers] + [c for c in swing_cands if c["ticker"] in day_tickers]

    picked: List[Dict] = []
    al3_used = False
    al3_sector: Optional[str] = None

    for c in swing_cands:
        if len(picked) >= SWING_MAX_FINAL:
            break

        al = int(c["al"])
        if al >= 3:
            if al3_used:
                continue  # AL3は1銘柄のみ
            al3_used = True
            al3_sector = str(c.get("sector", ""))

        # AL3がある場合、他はAL1まで（同一セクター暴走/心理暴走対策）
        if al3_used and al < 3:
            c = c.copy()
            if int(c["al"]) > 1:
                c["al"] = 1
                c["rec_lev"] = min(float(c["rec_lev"]), adjust_leverage_by_market(leverage_from_al(1), 50))

        picked.append(c)

    return picked


# ============================================================
# Dayスクリーニング
# ============================================================
def run_day(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return []

    uni = filter_earnings(uni, today_date)

    out: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        hist_d = fetch_history(ticker, period="180d")
        if hist_d is None or len(hist_d) < 80:
            continue

        day_score = score_daytrade_candidate(hist_d, mkt_score=mkt_score)
        if not np.isfinite(day_score) or day_score < DAY_SCORE_MIN:
            continue

        hist_i = fetch_intraday(ticker, period="5d", interval="5m")
        if hist_i is None or len(hist_i) < 50:
            continue

        rr_info = compute_tp_sl_rr(hist_d, mkt_score=mkt_score, for_day=True)
        rr = float(rr_info["rr"])
        if rr < DAY_RR_MIN:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        price_now = _safe_float(hist_i["Close"].iloc[-1], np.nan)
        entry = float(rr_info["entry"])

        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        out.append(dict(
            ticker=ticker,
            name=name,
            sector=sector,
            score=float(day_score),
            rr=float(rr),
            entry=float(entry),
            price_now=float(price_now) if np.isfinite(price_now) else np.nan,
            gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
            tp_pct=float(rr_info["tp_pct"]),
            sl_pct=float(rr_info["sl_pct"]),
            tp_price=float(rr_info["tp_price"]),
            sl_price=float(rr_info["sl_price"]),
            entry_basis=str(rr_info.get("entry_basis", "day")),
        ))

    out.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    return out[:DAY_MAX_FINAL]


# ============================================================
# レバレッジ（全体）
# ============================================================
def recommend_leverage_overall(mkt_score: int, swing_picks: List[Dict]) -> Tuple[float, str]:
    """
    ②：日次レバは「最大の良い押し目（Swing）」に引っ張られる。
    Swingが無い日は、地合いベースで控えめ。
    """
    if swing_picks:
        lev = float(max([float(c.get("rec_lev", 1.0)) for c in swing_picks] + [1.0]))
        # コメント
        al_max = int(max([int(c.get("al", 0)) for c in swing_picks] + [0]))
        if al_max >= 3:
            cmt = "攻め（押し目優位：AL3）"
        elif al_max == 2:
            cmt = "通常（押し目優位：AL2）"
        else:
            cmt = "軽め（AL1）"
        return float(np.clip(lev, 1.0, LEV_MAX)), cmt

    # Swing無し：地合いのみで控えめ（Day回転は別枠で考える）
    if mkt_score >= 70:
        return 1.8, "強め（Swing不在）"
    if mkt_score >= 60:
        return 1.5, "やや強め（Swing不在）"
    if mkt_score >= 50:
        return 1.3, "中立（Swing不在）"
    if mkt_score >= 40:
        return 1.1, "弱め（Swing不在）"
    return 1.0, "弱い（Swing不在）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# レポート
# ============================================================
def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str: str, today_date, mkt: Dict,
                 pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events = build_event_warnings(today_date)

    day = run_day(today_date, mkt_score)
    swing_cands = run_swing_candidates(today_date, mkt_score)
    swing = apply_swing_constraints(swing_cands, day)

    # 同一銘柄：DayとSwingは併用禁止（Swing優先）
    swing_tickers = {c["ticker"] for c in swing}
    day = [c for c in day if c["ticker"] not in swing_tickers]

    # ②のNO-TRADE（緩和）：EV不足日にだけ新規禁止（Swing基準）
    no_trade = False
    reasons: List[str] = []
    if not swing:
        no_trade = True
        reasons.append("Swing該当なし（AL>=1 が0）")
    else:
        avg_ev = float(np.mean([c["ev_r"] for c in swing]))
        has_strong = any(c["in_rank"] == "強IN" for c in swing)
        has_rr2 = any(float(c["rr"]) >= 2.0 for c in swing)
        if (avg_ev < 0.25) and (not has_strong) and (not has_rr2):
            no_trade = True
            reasons.append("Swing平均EV<0.25R かつ 強INなし かつ RR>=2.0なし")

    lev, lev_comment = recommend_leverage_overall(mkt_score, swing if not no_trade else [])
    max_pos = calc_max_position(total_asset, lev)

    # 全体の平均値
    swing_avg_rr = float(np.mean([c["rr"] for c in swing])) if swing else np.nan
    swing_avg_ev = float(np.mean([c["ev_r"] for c in swing])) if swing else np.nan
    day_avg_rr = float(np.mean([c["rr"] for c in day])) if day else np.nan

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（② Conditional Aggression）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors):
            lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    # NO-TRADE（緩和版）
    if no_trade:
        lines.append("🚫 本日は新規見送り（②：EV不足日）")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")
        lines.append("🏆 Swing（数日〜2週）")
        lines.append("- 本日は新規見送り")
        lines.append("")
        lines.append("⚡ Day（デイトレ）")
        lines.append("- 本日は新規見送り")
        lines.append("")
    else:
        # --- SWING ---
        lines.append("🏆 Swing（数日〜2週）Core候補（②：押し目良ければ攻める）")
        if swing:
            lines.append(
                f"  候補数:{len(swing)}銘柄 / 平均RR:{swing_avg_rr:.2f}R / 平均EV:{swing_avg_ev:.2f}R"
            )
            lines.append("")
            for c in swing:
                lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
                lines.append(
                    f"  AL:{c['al']} 推奨レバ:{float(c['rec_lev']):.1f}x  Score:{c['score']:.1f}  IN:{c['in_rank']}"
                )
                lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R")
                if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                    lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
                else:
                    lines.append(f"  押し目基準IN:{c['entry']:.1f}")
                lines.append(
                    f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})"
                )
                lines.append("")
        else:
            lines.append("- 該当なし")
            lines.append("")

        # --- DAY ---
        lines.append("⚡ Day（デイトレ）候補（併用禁止：Swing採用銘柄は除外）")
        if day:
            lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{day_avg_rr:.2f}R")
            lines.append("")
            for c in day:
                lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
                lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
                if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                    lines.append(f"  Day基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
                else:
                    lines.append(f"  Day基準IN:{c['entry']:.1f}")
                lines.append(
                    f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})"
                )
                lines.append("")
        else:
            lines.append("- 該当なし")
            lines.append("")

    # --- POS ---
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（json={"text": ...}）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return

    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE RESULT]", r.status_code, str(r.text)[:200])


# ============================================================
# Main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
