from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

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

EARNINGS_EXCLUDE_DAYS = 3

# Swing
SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.40

# Day
DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.5

# 表示
SECTOR_TOP_N = 5

# v11.1 必須（NO-TRADE）
NO_TRADE_MKT_SCORE_TH = int(os.getenv("NO_TRADE_MKT_SCORE_TH", "45"))
NO_TRADE_SWING_AVG_EV_TH = float(os.getenv("NO_TRADE_SWING_AVG_EV_TH", "0.3"))

# 追加カード
# 1) Day候補が0〜1のときだけ、Dayセクター→Swing制限を解除
DAY_COUNT_FOR_SECTOR_LIMIT = int(os.getenv("DAY_COUNT_FOR_SECTOR_LIMIT", "2"))  # 2以上で制限ON（=0-1は解除）
SECTOR_SWING_LIMIT_IF_DAY_PRESENT = int(os.getenv("SECTOR_SWING_LIMIT_IF_DAY_PRESENT", "1"))

# 2) 指数GU時 Day全停止（寄り後の実ギャップで判定）
INDEX_GU_STOP_ATR_TH = float(os.getenv("INDEX_GU_STOP_ATR_TH", "1.2"))
INDEX_GU_STOP_PCT_TH = float(os.getenv("INDEX_GU_STOP_PCT_TH", "1.2"))  # %（目安）
INDEX_GU_SYMBOLS = os.getenv("INDEX_GU_SYMBOLS", "^N225,^TOPX").split(",")

# 3) イベント当日 Day完全停止
STOP_DAY_ON_EVENT_TODAY = os.getenv("STOP_DAY_ON_EVENT_TODAY", "1") == "1"


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
    has_event_today = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if delta == 0:
            has_event_today = True

        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns, has_event_today


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
# インデックスGU判定（寄り後Day停止用）
# ============================================================
def _atr_from_hist(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else np.nan


def index_gu_stop_reason() -> Optional[str]:
    """
    指数の当日Openと前日Closeのギャップが大きい → Day全停止
    ※ ログ無し前提で「荒れ相場の朝」を構造的に回避するため
    """
    for sym in [s.strip() for s in INDEX_GU_SYMBOLS if s.strip()]:
        try:
            h = yf.Ticker(sym).history(period="30d", auto_adjust=True)
            if h is None or h.empty or len(h) < 20:
                continue
            # 当日分が取れてない場合があるため、最後2本を使う（最新=当日 or 前日）
            h = h.dropna()
            if len(h) < 2:
                continue

            atr = _atr_from_hist(h, 14)
            if not (np.isfinite(atr) and atr > 0):
                continue

            o = float(h["Open"].iloc[-1])
            c_prev = float(h["Close"].iloc[-2])
            gap = o - c_prev
            gap_atr = abs(gap) / atr
            gap_pct = abs(gap) / c_prev * 100.0 if c_prev > 0 else 0.0

            if gap_atr >= INDEX_GU_STOP_ATR_TH or gap_pct >= INDEX_GU_STOP_PCT_TH:
                return f"指数GU大（{sym} gap {gap_pct:.2f}% / {gap_atr:.2f}ATR）→ Day停止"
        except Exception:
            continue
    return None


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
# Swingスクリーニング（既存骨格維持 + v11.1排除はscoring/rrで担保）
# ============================================================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
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

        base_score = score_stock(ticker=ticker, df=hist, meta=row.to_dict(), run_mode="preopen", cfg={})
        if base_score is None:
            continue
        # score_stock(v11.1) の total_score / rr_adj / ev_r_adj を使う（理論RR排除も含む）
        total_score = float(base_score.get("total_score", 0.0))
        rr_adj = float(base_score.get("rr_adj", 0.0))
        ev_adj = float(base_score.get("ev_r_adj", 0.0))

        if not (np.isfinite(total_score) and total_score >= MIN_SCORE):
            continue
        if rr_adj < RR_MIN:
            continue
        if ev_adj < EV_MIN:
            continue

        # INランク（既存の思想を尊重）
        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue
        if mkt_score <= 45 and in_rank == "弱めIN":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info["rr"])

        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        entry = float(rr_info["entry"])
        gap_pct = (price_now / entry - 1.0) * 100.0 if np.isfinite(price_now) and entry > 0 else np.nan

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(total_score),
                rr=float(rr),
                ev_r=float(ev_adj),
                in_rank=in_rank,
                entry=entry,
                entry_basis=str(rr_info.get("entry_basis", "pullback")),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
            )
        )

    cands.sort(key=lambda x: (x["score"], x["ev_r"], x["rr"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# Dayスクリーニング（寄り後再判定=押し戻し→再上昇をscore_daytrade_candidateに反映）
# ============================================================
def run_day(today_date, mkt_score: int, stop_reason: Optional[str]) -> Tuple[List[Dict], Optional[str]]:
    if stop_reason:
        return [], stop_reason

    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], None

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], None

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

        rr_info = compute_tp_sl_rr(hist_d, mkt_score=mkt_score, for_day=True)
        rr = float(rr_info["rr"])
        if rr < DAY_RR_MIN:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        out.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(day_score),
                rr=rr,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
                entry=float(rr_info["entry"]),
                entry_basis=str(rr_info.get("entry_basis", "day")),
            )
        )

    out.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    return out[:DAY_MAX_FINAL], None


# ============================================================
# Swing × Day 競合制御（カード①の条件付き解除つき）
# ============================================================
def apply_conflict_controls(swing: List[Dict], day: List[Dict]) -> Tuple[List[Dict], List[str]]:
    warns: List[str] = []
    if not swing:
        return swing, warns

    day_tickers = {d["ticker"] for d in day} if day else set()
    if day_tickers:
        before = len(swing)
        swing = [s for s in swing if s["ticker"] not in day_tickers]
        if len(swing) != before:
            warns.append("⚠ 同一銘柄：DayとSwingの併用禁止（重複は除外）")

    # Day候補が0〜1なら、セクター制限解除
    if day and len(day) >= DAY_COUNT_FOR_SECTOR_LIMIT:
        day_sectors = {d.get("sector", "") for d in day if d.get("sector", "")}
        if day_sectors:
            kept: List[Dict] = []
            dropped = 0
            for sec in day_sectors:
                sec_swing = [s for s in swing if s.get("sector", "") == sec]
                if len(sec_swing) <= SECTOR_SWING_LIMIT_IF_DAY_PRESENT:
                    continue
                # EV→RR→Scoreで残す
                sec_swing_sorted = sorted(sec_swing, key=lambda x: (x.get("ev_r", 0), x.get("rr", 0), x.get("score", 0)), reverse=True)
                keep_set = {x["ticker"] for x in sec_swing_sorted[:SECTOR_SWING_LIMIT_IF_DAY_PRESENT]}
                for s in swing:
                    if s.get("sector", "") == sec and s["ticker"] not in keep_set:
                        dropped += 1
            if dropped > 0:
                swing = [s for s in swing if not (s.get("sector", "") in day_sectors and any(s["ticker"] == x["ticker"] for x in swing))]
                # 上の行は意図せず全消しリスクあるので再構成（確実）
                new_swing = []
                for sec in day_sectors:
                    sec_swing = [s for s in swing if s.get("sector", "") == sec]
                    sec_swing_sorted = sorted(sec_swing, key=lambda x: (x.get("ev_r", 0), x.get("rr", 0), x.get("score", 0)), reverse=True)
                    new_swing.extend(sec_swing_sorted[:SECTOR_SWING_LIMIT_IF_DAY_PRESENT])
                other = [s for s in swing if s.get("sector", "") not in day_sectors]
                swing = sorted(other + new_swing, key=lambda x: (x.get("score", 0), x.get("ev_r", 0), x.get("rr", 0)), reverse=True)[:SWING_MAX_FINAL]
                warns.append("⚠ 同セクターDayあり → Swingは1銘柄制限（Dayが2銘柄以上のときのみ）")

    return swing, warns


# ============================================================
# NO-TRADE DAY
# ============================================================
def decide_no_trade_day(mkt_score: int, swing: List[Dict], day: List[Dict]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if mkt_score < NO_TRADE_MKT_SCORE_TH:
        reasons.append(f"mkt_score < {NO_TRADE_MKT_SCORE_TH}")

    if swing:
        avg_ev = float(np.mean([_safe_float(c.get("ev_r", np.nan), np.nan) for c in swing]))
        if not np.isfinite(avg_ev) or avg_ev < NO_TRADE_SWING_AVG_EV_TH:
            reasons.append(f"Swing候補の平均EV < {NO_TRADE_SWING_AVG_EV_TH:.1f}R")
    else:
        reasons.append("Swing候補なし")

    # DayはGU危険域判定を持っていないため「全停止理由」は別カードで実施（指数GU/イベント）
    return (len(reasons) > 0), reasons


# ============================================================
# レバレッジ
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


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

    lev, lev_comment = recommend_leverage(mkt_score)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events, has_event_today = build_event_warnings(today_date)

    # --- Day停止理由（カード②③） ---
    day_stop_reason = index_gu_stop_reason()
    if STOP_DAY_ON_EVENT_TODAY and has_event_today:
        day_stop_reason = "イベント当日 → Day停止"

    swing = run_swing(today_date, mkt_score)
    day, day_stop_reason2 = run_day(today_date, mkt_score, day_stop_reason)
    if day_stop_reason2:
        day_stop_reason = day_stop_reason2

    # Swing × Day 競合制御（カード①）
    swing, conflict_warns = apply_conflict_controls(swing, day)

    # NO-TRADE DAY（カード必須）
    no_trade, no_trade_reasons = decide_no_trade_day(mkt_score, swing, day)

    # Swing平均EV（表示用）
    swing_avg_ev = float(np.mean([c["ev_r"] for c in swing])) if swing else 0.0

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報 (v11.1+)")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    if no_trade:
        lines.append("🚫 本日は新規見送り日（条件該当）")
        for r in no_trade_reasons:
            lines.append(f"- {r}")
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

    if day_stop_reason:
        lines.append(f"🛑 Day停止: {day_stop_reason}")
        lines.append("")

    if conflict_warns:
        lines.extend(conflict_warns)
        lines.append("")

    # --- SWING ---
    lines.append("🏆 Swing（数日〜2週）Core候補")
    if swing:
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均EV:{swing_avg_ev:.2f}R")
        lines.append("")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']} EV:{c['ev_r']:.2f}R")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  押し目基準IN:{c['entry']:.1f}")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # --- DAY ---
    lines.append("⚡ Day（デイトレ）候補")
    if day:
        rr_vals = [c["rr"] for c in day]
        avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0
        lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{avg_rr:.2f}R")
        lines.append("")
        for c in day:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # --- POS ---
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    if no_trade:
        lines.append("")
        lines.append("✅ ルール: 本日は新規エントリー禁止（監視のみ）")
    elif day_stop_reason:
        lines.append("")
        lines.append("✅ ルール: Day停止（Swingのみ条件一致で）")
    else:
        lines.append("")
        lines.append("✅ ルール: 条件一致のみ（無理はしない）")

    return "\n".join(lines)


# ============================================================
# LINE送信（届く形を厳守：json={"text": ...}）
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
