from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


# ============================================================
# 設定（Swing専用 / LINE通知）
# - LINE送信方式は「届く」版（json={"text": ...}）を維持
# - デイは完全に削除
# - 逆張りを混ぜない：TrendGate 必須
# - 地合いでスクリーニング基準を可変にしない（普段から勝てる基準）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外（universe_jpx.csv に earnings_date がある場合のみ）
EARNINGS_EXCLUDE_DAYS = 3

# Swing（固定基準）
SWING_MAX_FINAL = 5
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40
REQUIRE_TREND_GATE = True

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
            time.sleep(0.5)
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


def build_event_warnings(today_date) -> List[str]:
    events = load_events()
    warns: List[str] = []

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns


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

    keep = []
    for d in parsed:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            keep.append(abs((d - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)

    return df[keep]


# ============================================================
# EV（R）
# ============================================================
def expected_r(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return float(win * rr - (1.0 - win) * 1.0)


# ============================================================
# Swing screening（順張り専用）
# ============================================================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # ticker列吸収
    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return []

    uni = filter_earnings(uni, today_date)

    cands: List[Dict] = []

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        # TrendGate（逆張り排除）
        if REQUIRE_TREND_GATE and (not trend_gate(hist)):
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc) or sc < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
        rr = float(rr_info.get("rr", 0.0))
        if rr < SWING_RR_MIN:
            continue

        ev = expected_r(in_rank, rr)
        if ev < SWING_EV_R_MIN:
            continue

        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        entry = float(rr_info.get("entry", 0.0))
        gap = np.nan
        if np.isfinite(price_now) and np.isfinite(entry) and entry > 0:
            gap = (price_now / entry - 1.0) * 100.0

        cands.append(
            dict(
                ticker=ticker,
                name=str(r.get("name", ticker)),
                sector=str(r.get("sector", r.get("industry_big", "不明"))),
                score=float(sc),
                in_rank=in_rank,
                rr=float(rr),
                ev=float(ev),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap) if np.isfinite(gap) else np.nan,
                tp_pct=float(rr_info.get("tp_pct", 0.0)),
                sl_pct=float(rr_info.get("sl_pct", 0.0)),
                tp_price=float(rr_info.get("tp_price", 0.0)),
                sl_price=float(rr_info.get("sl_price", 0.0)),
            )
        )

    # ソート：Score → EV → RR
    cands.sort(key=lambda x: (x["score"], x["ev"], x["rr"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# レポート
# ============================================================
def _fmt_money(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}円"
    except Exception:
        return "0円"


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    # レバは地合いを使ってOK（スクリーニングは固定基準）
    lev = 1.3 if mkt_score < 50 else 1.7
    max_pos = float(total_asset) * lev

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events = build_event_warnings(today_date)
    swing = run_swing(today_date, mkt_score=mkt_score)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{_fmt_money(max_pos)}")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s, p) in enumerate(sectors, 1):
            lines.append(f"{i}. {s} ({p:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(events)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ）")
    if swing:
        rr_vals = [c["rr"] for c in swing if np.isfinite(c["rr"])]
        ev_vals = [c["ev"] for c in swing if np.isfinite(c["ev"])]
        if rr_vals and ev_vals:
            lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f} / 平均EV:{float(np.mean(ev_vals)):.2f}")
            lines.append("")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f} EV:{c['ev']:.2f} IN:{c['in_rank']}")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"  IN:{c['entry']:.1f} 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  IN:{c['entry']:.1f}")
            lines.append(f"  TP:{c['tp_price']:.1f} SL:{c['sl_price']:.1f}")
            lines.append("")
    else:
        lines.append("- 該当なし（順張り条件に合う押し目が無い）")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（届く版 + リトライ）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL not set. Printing report only.")
        print(text)
        return

    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    headers = {"Content-Type": "application/json", "User-Agent": "stockbotTOM/1.0 (+github-actions)"}

    for idx, ch in enumerate(chunks, start=1):
        last_err = None
        for attempt in range(1, 4):
            try:
                r = requests.post(WORKER_URL, json={"text": ch}, headers=headers, timeout=20)
                print("[LINE RESULT]", f"chunk={idx}/{len(chunks)}", f"attempt={attempt}", r.status_code, str(r.text)[:200])
                if 200 <= r.status_code < 300:
                    last_err = None
                    break
                last_err = f"HTTP {r.status_code}: {str(r.text)[:200]}"
                time.sleep(0.8 * attempt)
            except Exception as e:
                last_err = repr(e)
                print("[LINE ERROR]", f"chunk={idx}/{len(chunks)}", f"attempt={attempt}", last_err)
                time.sleep(0.8 * attempt)

        if last_err:
            print("[LINE FAIL]", last_err)


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

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
