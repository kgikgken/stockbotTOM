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
from utils.scoring import calc_inout_for_stock, score_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.day import score_daytrade_candidate
from utils.qualify import qualify_runner_grade


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外
EARNINGS_EXCLUDE_DAYS = 3

# Swing（vAB_prime）
SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.40

# Day（追い禁止 + 実効RR）
DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.2
DAY_RR_EFF_MIN = 1.10  # rr*0.70 の床
DAY_CHASE_GU_MAX_PCT = 2.5  # entry比 +2.5%超は追い禁止扱い

# 表示
SECTOR_TOP_N = 5


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
            time.sleep(0.35)
    return None


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.35)
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
    events = load_events()
    warns: List[str] = []
    is_near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            is_near = True
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns, is_near


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
# EV(R) - 暫定
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
# Conditional Aggression レバ
# ============================================================
def recommend_leverage(mkt_score: int, al: int, event_near: bool) -> Tuple[float, str]:
    # イベント近接は上限を落とす（暴発防止）
    if event_near:
        if al >= 3:
            return 2.0, "攻め（AL3一点のみ）/イベント近接で2.0x上限"
        return 1.3, "守り（イベント近接）"

    # 通常日
    if al >= 3:
        if mkt_score >= 55:
            return 2.3, "攻め（押し目優位：AL3）"
        if mkt_score >= 45:
            return 2.0, "攻め（押し目優位：AL3）/地合い弱めで2.0x"
        return 1.7, "やや攻め（AL3だが地合い弱い）"

    if al == 2:
        if mkt_score >= 60:
            return 1.7, "やや強気（AL2）"
        if mkt_score >= 50:
            return 1.5, "中立（AL2）"
        return 1.3, "守り（AL2）"

    # al == 1
    if mkt_score >= 65:
        return 1.5, "中立（地合い良）"
    if mkt_score >= 50:
        return 1.3, "中立"
    return 1.1, "守り"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# Swing vAB Prime（Runner→押し目）
# ============================================================
def run_swing(today_date, mkt_score: int, event_near: bool) -> List[Dict]:
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

    cands: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker, period="320d")
        if hist is None or len(hist) < 140:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score) or base_score < SWING_SCORE_MIN:
            continue

        # vAB_prime: Runner判定（A1/A2）
        q = qualify_runner_grade(hist)
        if not q["is_runner"]:
            continue

        # 押し目（INランク）
        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        # RR（構造）
        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
        rr = float(rr_info["rr"])
        if rr < SWING_RR_MIN:
            continue

        ev_r = expected_r_from_in_rank(in_rank, rr)
        if ev_r < SWING_EV_R_MIN:
            continue

        # AL（攻めレベル）: Runner grade × INランク
        al = int(q["grade"])
        if in_rank == "強IN":
            al = min(3, al + 1)
        elif in_rank == "弱めIN":
            al = max(1, al - 1)

        # イベント近接はAL3一点のみ
        if event_near and al < 3:
            continue

        lev, lev_comment = recommend_leverage(mkt_score, al=al, event_near=event_near)

        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        entry = float(rr_info["entry"])
        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry):
            gap_pct = (price_now / entry - 1.0) * 100.0

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(base_score),
                rr=float(rr),
                ev_r=float(ev_r),
                in_rank=in_rank,
                al=int(al),
                lev=float(lev),
                lev_comment=str(lev_comment),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
                runner_kind=str(q["kind"]),
                runner_strength=float(q["strength"]),
            )
        )

    # ソート: AL → EV → Score → RR
    cands.sort(key=lambda x: (x["al"], x["ev_r"], x["score"], x["rr"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# Day（追い禁止 + 実効RR床）
# ============================================================
def run_day(today_date, mkt_score: int, swing_picks: List[str]) -> List[Dict]:
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
    swing_set = set(swing_picks)

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker or ticker in swing_set:
            continue

        hist_d = fetch_history(ticker, period="200d")
        if hist_d is None or len(hist_d) < 90:
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

        entry = float(rr_info["entry"])
        price_now = _safe_float(hist_i["Close"].iloc[-1], np.nan)

        # 追い禁止（GU危険域）
        if np.isfinite(price_now) and entry > 0:
            chase_pct = (price_now / entry - 1.0) * 100.0
            if chase_pct > DAY_CHASE_GU_MAX_PCT:
                continue

        # 実効RR（スリッページ・取りこぼし込み）
        rr_eff = rr * 0.70
        if rr_eff < DAY_RR_EFF_MIN:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry):
            gap_pct = (price_now / entry - 1.0) * 100.0

        out.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(day_score),
                rr=float(rr),
                rr_eff=float(rr_eff),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
            )
        )

    out.sort(key=lambda x: (x["score"], x["rr_eff"], x["rr"]), reverse=True)
    return out[:DAY_MAX_FINAL]


# ============================================================
# レポート
# ============================================================
def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    events, event_near = build_event_warnings(today_date)
    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)

    swing = run_swing(today_date, mkt_score, event_near=event_near)
    swing_picks = [c["ticker"] for c in swing]
    day = run_day(today_date, mkt_score, swing_picks=swing_picks)

    # 今日の推奨レバ（全体）：Swingの最上位ALに合わせる（無ければ守り）
    top_al = max([c["al"] for c in swing], default=1)
    lev, lev_comment = recommend_leverage(mkt_score, al=top_al, event_near=event_near)
    max_pos = calc_max_position(total_asset, lev)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（vAB_prime / 大勝ちモード）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append(f"- イベント近接: {'YES' if event_near else 'NO'}")
    if event_near:
        lines.append("補足: イベント近接→AL3一点のみ許可。")
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

    # --- SWING ---
    lines.append("🏆 Swing（数日〜2週）Core候補（vAB_prime：Runner→押し目）")
    if swing:
        rr_vals = [c["rr"] for c in swing]
        ev_vals = [c["ev_r"] for c in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f}R / 平均EV:{float(np.mean(ev_vals)):.2f}R")
        lines.append("")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  AL:{c['al']} 推奨レバ:{c['lev']:.1f}x  Score:{c['score']:.1f}  IN:{c['in_rank']}  Runner:{c['runner_kind']}")
            lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R  走行強度:{c['runner_strength']:.1f}")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  押し目基準IN:{c['entry']:.1f}")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし（vAB_primeは“走る銘柄だけ”残す）")
        lines.append("")

    # --- DAY ---
    lines.append("⚡ Day（デイトレ）候補（追い禁止 + 実効RR床 / Swing採用銘柄は除外）")
    if day:
        rr_vals = [c["rr"] for c in day]
        rr_eff_vals = [c["rr_eff"] for c in day]
        lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f}R（実効:{float(np.mean(rr_eff_vals)):.2f}R）")
        lines.append("")
        for c in day:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R（実効:{c['rr_eff']:.2f}R）")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
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
