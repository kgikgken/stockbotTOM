from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score, abnormal_day_flag
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.day import score_daytrade_candidate
from utils.risk import (
    sl_cluster_filter,
    entry_unreachable,
    count_resistance_pivots,
    rr_quality_ok,
    load_cooldown,
    save_cooldown,
    cooldown_ok,
    update_cooldown_if_tp_hit,
    downgrade_al,
)


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外
EARNINGS_EXCLUDE_DAYS = 3

# Swing
SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.30

# Day
DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.5
DAY_EFFECTIVE_RR_MULT = 0.70  # 改善④

# 表示
SECTOR_TOP_N = 5

# Conditional Aggression
LEV_MAX = 2.5
LEV_CAP_WEAK_MKT = 2.0  # mkt_score < 50 で全体推奨レバを抑える

# ⑥〜⑩
SL_CLUSTER_TOL = 0.003     # 0.3%
ENTRY_MAX_GAP = 0.015      # +1.5% 超は追わない
MAX_RESISTANCE = 2
COOLDOWN_DAYS = 3
COOLDOWN_PATH = "cooldown_tp.csv"


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


def build_event_warnings(today_date) -> Tuple[List[str], bool]:
    """
    戻り: (warnings, is_event_caution_day)
    caution: -1〜+2日
    """
    warns: List[str] = []
    caution = False

    for ev in load_events():
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            caution = True
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    return (warns or ["- 特になし"]), caution


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
# Conditional Aggression（AL）
# ============================================================
def calc_aggression_level(in_rank: str, rr: float, ev_r: float) -> int:
    if in_rank == "強IN" and rr >= 2.8 and ev_r >= 0.6:
        return 3
    if in_rank in ("強IN", "通常IN") and rr >= 2.2 and ev_r >= 0.4:
        return 2
    if in_rank == "通常IN" and rr >= 1.8 and ev_r >= 0.3:
        return 1
    return 0


def leverage_from_al(al: int) -> float:
    if al >= 3:
        return 2.3
    if al == 2:
        return 1.7
    if al == 1:
        return 1.3
    return 1.0


def adjust_leverage_by_market(base_lev: float, mkt_score: int) -> float:
    lev = float(base_lev)
    if mkt_score < 40:
        lev -= 0.35
    elif mkt_score < 45:
        lev -= 0.20
    elif mkt_score >= 70:
        lev += 0.10
    return float(np.clip(lev, 1.0, LEV_MAX))


# ============================================================
# Swing候補生成（フィルタ込）
# ============================================================
def run_swing_candidates(today_date, mkt_score: int, cooldown_map: Dict,
                         abnormal_steps: int, event_caution: bool) -> Tuple[List[Dict], Dict]:
    """
    戻り: (candidates, cooldown_map_updated)
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], cooldown_map

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], cooldown_map

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

        # ⑨ cooldown
        if not cooldown_ok(ticker, today_date, cooldown_map, cooldown_days=COOLDOWN_DAYS):
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

        entry = float(rr_info["entry"])
        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        tp_price = float(rr_info["tp_price"])

        # ⑨ TP到達記録（近似）
        update_cooldown_if_tp_hit(ticker, today_date, float(price_now), tp_price, cooldown_map)

        # ⑦ Entry未到達の自動見送り（追わない）
        if entry_unreachable(entry, float(price_now), max_gap=ENTRY_MAX_GAP):
            continue

        # ⑧ RR質フィルタ（抵抗帯数）
        res_cnt = count_resistance_pivots(hist["Close"], entry, tp_price, window=2)
        if not rr_quality_ok(rr, res_cnt, max_res=MAX_RESISTANCE):
            continue

        al = calc_aggression_level(in_rank, rr, ev_r)

        # ⑩ 異常日：ALを1段階デグレード
        if abnormal_steps > 0:
            al = downgrade_al(al, steps=abnormal_steps)

        if al <= 0:
            continue

        # ③ イベント警戒日：AL3のみ許可
        if event_caution and al < 3:
            continue

        base_lev = leverage_from_al(al)
        rec_lev = adjust_leverage_by_market(base_lev, mkt_score)

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
            tp_price=float(tp_price),
            sl_price=float(rr_info["sl_price"]),
            resistance_count=int(res_cnt),
        ))

    # 優先：AL → Score → EV → RR
    cands.sort(key=lambda x: (x["al"], x["score"], x["ev_r"], x["rr"]), reverse=True)
    return cands, cooldown_map


def apply_swing_constraints(swing_cands: List[Dict]) -> List[Dict]:
    """
    改善①②⑥:
    - AL3は最大1銘柄
    - AL3がある日: AL2=0、AL1最大1（一点集中）
    - AL3日のAL1は EV>=0.35 必須
    - SLクラスタ（⑥）
    """
    if not swing_cands:
        return []

    # ⑥ SLクラスタ（まず粗く間引く）
    swing_cands = sl_cluster_filter(swing_cands, tol=SL_CLUSTER_TOL)

    picked: List[Dict] = []
    al3 = None

    # 1) まずAL3を1つ取る（あれば）
    for c in swing_cands:
        if int(c["al"]) >= 3:
            al3 = c
            picked.append(c)
            break

    # 2) AL3がある日：AL1を最大1（EV>=0.35）、AL2は0
    if al3 is not None:
        for c in swing_cands:
            if c["ticker"] == al3["ticker"]:
                continue
            if int(c["al"]) == 1 and float(c.get("ev_r", 0.0)) >= 0.35:
                picked.append(c)
                break
        return picked[:2]  # AL3 + AL1(max1)

    # 3) AL3が無い日：通常の上位から最大3
    for c in swing_cands:
        if len(picked) >= SWING_MAX_FINAL:
            break
        picked.append(c)

    return picked[:SWING_MAX_FINAL]


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
            eff_rr=float(rr) * DAY_EFFECTIVE_RR_MULT,
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
# 全体推奨レバ
# ============================================================
def recommend_leverage_overall(mkt_score: int, swing_picks: List[Dict]) -> Tuple[float, str]:
    if swing_picks:
        lev = float(max([float(c.get("rec_lev", 1.0)) for c in swing_picks] + [1.0]))
        al_max = int(max([int(c.get("al", 0)) for c in swing_picks] + [0]))
        if al_max >= 3:
            cmt = "攻め（押し目優位：AL3）"
        elif al_max == 2:
            cmt = "通常（押し目優位：AL2）"
        else:
            cmt = "軽め（AL1）"
        # 全体キャップ（地合い弱い日は踏み過ぎ防止）
        if mkt_score < 50:
            lev = min(lev, LEV_CAP_WEAK_MKT)
            cmt += f" / 地合い<50で{LEV_CAP_WEAK_MKT:.1f}x上限"
        return float(np.clip(lev, 1.0, LEV_MAX)), cmt

    # Swing無し：控えめ
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


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events, event_caution = build_event_warnings(today_date)

    abnormal = abnormal_day_flag()
    abnormal_steps = 1 if abnormal.flag else 0

    # cooldown
    cooldown_map = load_cooldown(COOLDOWN_PATH)

    day = run_day(today_date, mkt_score)

    swing_cands, cooldown_map = run_swing_candidates(
        today_date=today_date,
        mkt_score=mkt_score,
        cooldown_map=cooldown_map,
        abnormal_steps=abnormal_steps,
        event_caution=event_caution,
    )

    swing = apply_swing_constraints(swing_cands)

    # 同一銘柄：DayとSwing併用禁止（Swing優先）
    swing_tickers = {c["ticker"] for c in swing}
    day = [c for c in day if c["ticker"] not in swing_tickers]

    # cooldown 保存（TP到達があった場合に更新済）
    save_cooldown(cooldown_map, COOLDOWN_PATH)

    # NO-TRADE（緩和版）
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

    lev, lev_comment = recommend_leverage_overall(mkt_score, [] if no_trade else swing)
    max_pos = calc_max_position(total_asset, lev)

    # ⑤ 行動理由1行
    action_line = "補足: "
    if abnormal.flag:
        action_line += "異常日フラグ→ALを1段階落として防御。"
    elif event_caution:
        action_line += "イベント近接→AL3一点のみ許可。"
    elif any(int(c.get("al", 0)) >= 3 for c in swing):
        action_line += "AL3あり→一点集中（AL3+AL1最大1）。"
    else:
        action_line += "押し目優位の範囲で通常運転。"

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（② Conditional Aggression / vUltimate）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    if abnormal.flag:
        lines.append(f"- 異常日: YES（ALデグレード）")
    else:
        lines.append(f"- 異常日: NO")
    lines.append(action_line)
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

    if abnormal.flag and abnormal.reasons:
        lines.append("🧯 異常日理由")
        for r in abnormal.reasons:
            lines.append(f"- {r}")
        lines.append("")

    # --- SWING ---
    if no_trade:
        lines.append("🚫 本日は新規見送り（緩和版: EV不足日）")
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
        lines.append("🏆 Swing（数日〜2週）Core候補（押し目良ければ攻める）")
        if swing:
            avg_rr = float(np.mean([c["rr"] for c in swing]))
            avg_ev = float(np.mean([c["ev_r"] for c in swing]))
            lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{avg_rr:.2f}R / 平均EV:{avg_ev:.2f}R")
            lines.append("")
            for c in swing:
                lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
                lines.append(f"  AL:{c['al']} 推奨レバ:{float(c['rec_lev']):.1f}x  Score:{c['score']:.1f}  IN:{c['in_rank']}")
                lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R  抵抗:{int(c.get('resistance_count', 0))}")
                if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                    lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
                else:
                    lines.append(f"  押し目基準IN:{c['entry']:.1f}")
                lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
                lines.append("")
        else:
            lines.append("- 該当なし")
            lines.append("")

        # --- DAY ---
        lines.append("⚡ Day（デイトレ）候補（Swing採用銘柄は除外）")
        if day:
            avg_rr = float(np.mean([c["rr"] for c in day]))
            avg_eff = float(np.mean([c["eff_rr"] for c in day]))
            lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{avg_rr:.2f}R（実効:{avg_eff:.2f}R）")
            lines.append("")
            for c in day:
                lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
                lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R（実効:{c['eff_rr']:.2f}R）")
                if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                    lines.append(f"  Day基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
                else:
                    lines.append(f"  Day基準IN:{c['entry']:.1f}")
                lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
                lines.append("")
        else:
            lines.append("- 該当なし")
            lines.append("")

    # --- POS ---
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
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


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
