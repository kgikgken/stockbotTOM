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
from utils.scoring import score_stock, calc_in_rank, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.qualify import runner_strength, runner_class, al3_score

# ============================================================
# 設定（機能は残す。Swing集中がデフォだが、将来 Day を戻せるようにスイッチだけ残す）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 出力
SECTOR_TOP_N = 5

# 決算フィルタ
EARNINGS_EXCLUDE_DAYS = 3

# Swing
SWING_TOP_N_DISPLAY = 5
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40
REQUIRE_TREND_GATE = True  # 逆張り排除（順張り徹底）
REQUIRE_RUNNER_MIN = 70.0  # 走行能力の床（上げたいならここを上げる）

# AL3のみ運用（大勝ちモード）
AL3_ONLY = True

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

def fetch_history(ticker: str, period: str = "320d") -> Optional[pd.DataFrame]:
    for _ in range(3):
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
    near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            near = True
            when = f"{delta}日後" if delta > 0 else ("本日" if delta == 0 else "直近")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns, near

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

    # 勝率仮定（ログで更新前提）
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
# Swing スクリーニング（順張り専用：TrendGate）
# ============================================================
def run_swing(today_date, mkt_score: int, event_near: bool) -> List[Dict]:
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

    # 決算回避
    uni = filter_earnings(uni, today_date)

    cands: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker, period="360d")
        if hist is None or len(hist) < 220:
            continue

        # Trend gate（逆張り排除）
        if REQUIRE_TREND_GATE:
            ok, _info = trend_gate(hist)
            if not ok:
                continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score) or base_score < SWING_SCORE_MIN:
            continue

        in_rank = calc_in_rank(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
        rr = float(rr_info["rr"])
        entry = float(rr_info["entry"])
        tp_pct = float(rr_info["tp_pct"])
        sl_pct = float(rr_info["sl_pct"])
        tp_price = float(rr_info["tp_price"])
        sl_price = float(rr_info["sl_price"])

        if rr < SWING_RR_MIN:
            continue

        ev_r = expected_r_from_in_rank(in_rank, rr)
        if ev_r < SWING_EV_R_MIN:
            continue

        # Runner / AL3
        r_strength = float(runner_strength(hist))
        r_class = runner_class(r_strength)
        if r_strength < REQUIRE_RUNNER_MIN:
            continue

        al3 = float(al3_score(float(base_score), rr, ev_r, r_strength))

        # event_near の日は AL3 だけに絞る（仕様：暴走防止）
        if event_near and AL3_ONLY:
            # keep but later we will cut to top1 if required in report stage
            pass

        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(base_score),
                rr=rr,
                ev_r=float(ev_r),
                in_rank=in_rank,
                entry=entry,
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                tp_price=tp_price,
                sl_price=sl_price,
                runner=r_class,
                runner_strength=float(r_strength),
                al3=float(al3),
            )
        )

    # Sort: AL3 -> EV -> RR -> Score
    cands.sort(key=lambda x: (x["al3"], x["ev_r"], x["rr"], x["score"]), reverse=True)

    # AL3_ONLY: show only top-N, but they are all AL3-quality already.
    return cands[: max(SWING_TOP_N_DISPLAY, 5)]

# ============================================================
# レバ
# ============================================================
def recommend_leverage(mkt_score: int, event_near: bool) -> Tuple[float, str]:
    # Swing集中 / AL3のみ：基本は攻めるが、地合い悪い日は上限を守る
    if mkt_score < 50:
        lev = 2.0
        comment = "AL3のみ（地合い<50で2.0x上限）"
    else:
        lev = 2.3
        comment = "AL3のみ（押し目が良いなら攻める）"
    if event_near:
        lev = min(lev, 2.0)
        comment += " / イベント近接で2.0x上限"
    return float(lev), comment

def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))

def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"

# ============================================================
# レポート
# ============================================================
def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    events, event_near = build_event_warnings(today_date)
    lev, lev_comment = recommend_leverage(mkt_score, event_near)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)

    swing = run_swing(today_date, mkt_score, event_near=event_near)

    # event_near の日は「最上位AL3一点」運用に寄せる（仕様）
    if event_near and swing:
        swing = swing[:1]

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing集中 / AL3 Top5）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append(f"- イベント近接: {'YES' if event_near else 'NO'}")
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

    lines.append("🏆 Swing（順張り/逆張り排除：TrendGate）")
    if swing:
        rr_vals = [c["rr"] for c in swing]
        ev_vals = [c["ev_r"] for c in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f}R / 平均EV:{float(np.mean(ev_vals)):.2f}R")
        lines.append("")
        for i, c in enumerate(swing, start=1):
            star = " ⭐" if i == 1 else ""
            lines.append(f"{i}. {c['ticker']} {c['name']}{star}")
            lines.append(f"   AL3:{c['al3']:.2f}  Score:{c['score']:.1f}  IN:{c['in_rank']}  Runner:{c['runner']}（走行強度:{c['runner_strength']:.1f}）")
            lines.append(f"   RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"   押し目IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"   押し目IN:{c['entry']:.1f}")
            lines.append(f"   TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし（順張り条件に合う押し目が無い）")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)

# ============================================================
# LINE送信（機能を落とさず、成功率を上げる）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL not set. Printing report only.")
        print(text)
        return

    # LINE notify っぽい長さ対策
    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "stockbotTOM/1.0 (+github-actions)",
    }

    for idx, ch in enumerate(chunks, start=1):
        payload = {"text": ch}
        last_err = None
        for attempt in range(1, 4):
            try:
                r = requests.post(WORKER_URL, json=payload, headers=headers, timeout=25)
                ok = (200 <= r.status_code < 300)
                print(f"[LINE] chunk {idx}/{len(chunks)} attempt {attempt} -> {r.status_code}")
                if ok:
                    break
                last_err = f"HTTP {r.status_code}: {str(r.text)[:200]}"
                time.sleep(0.8 * attempt)
            except Exception as e:
                last_err = repr(e)
                print(f"[LINE] chunk {idx}/{len(chunks)} attempt {attempt} EXC: {last_err}")
                time.sleep(0.8 * attempt)

        if last_err and "HTTP" in last_err:
            print("[LINE] last_err:", last_err)

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
