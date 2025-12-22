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
from utils.scoring import score_stock, calc_inout_for_stock, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


# ============================================================
# 設定（Swing専用 / LINEに届く仕様を維持）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 出力
SECTOR_TOP_N = 5

# 決算フィルタ
EARNINGS_EXCLUDE_DAYS = 3

# Swing（"勝てる形"だけを濃くする：スコアは表示しない）
SWING_MAX_FINAL = 5

# 内部足切り（表示しない）
SCORE_FLOOR = 60.0

# コア条件
SWING_RR_MIN = 2.5
EV_MIN = 0.80
EV_MAX = 2.50
OFF_HIGH_MIN = -10.0
OFF_HIGH_MAX = -3.0
MAX_GAP_PCT = 3.0

# TPタッチ実績（直近N日で1回以上）
TP_HIT_LOOKBACK = 60
TP_HIT_MIN = 1


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


def fetch_history(ticker: str, period: str = "300d") -> Optional[pd.DataFrame]:
    for _ in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.5)
    return None


def _tp_hit_count(hist: pd.DataFrame, tp_price: float, lookback: int = 60) -> int:
    if hist is None or hist.empty or not np.isfinite(tp_price) or tp_price <= 0:
        return 0
    col = "High" if "High" in hist.columns else "Close"
    s = hist[col].astype(float).tail(int(max(5, lookback)))
    if s.empty:
        return 0
    return int(np.sum(s.values >= float(tp_price)))


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
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

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
# EV（勝率は固定仮定：フィルタ/順位づけ用途）
# ============================================================
def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return float(win * rr - (1.0 - win) * 1.0)


# ============================================================
# Swing スクリーニング（順張り専用）
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
        if hist is None or len(hist) < 120:
            continue

        # Trend gate（逆張り排除）
        if not trend_gate(hist):
            continue

        # 内部足切り（表示しない）
        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc) or sc < SCORE_FLOOR:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank not in ("強IN", "通常IN"):
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)

        rr = float(rr_info["rr"])
        entry = float(rr_info["entry"])
        tp_price = float(rr_info["tp_price"])
        sl_price = float(rr_info["sl_price"])

        if rr < SWING_RR_MIN:
            continue

        # 現在値 & GAP（追いかけ禁止）
        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        if not (np.isfinite(price_now) and np.isfinite(entry) and entry > 0):
            continue
        gap_pct = (price_now / entry - 1.0) * 100.0
        if gap_pct > MAX_GAP_PCT:
            continue

        # 直近高値からの距離（%）: calc_inoutで使ってるが念のため再チェック
        try:
            c = hist["Close"].astype(float)
            high_60 = float(c.tail(60).max())
            off_high = (float(c.iloc[-1]) - high_60) / (high_60 + 1e-9) * 100.0
        except Exception:
            off_high = np.nan
        if not (np.isfinite(off_high) and OFF_HIGH_MIN <= off_high <= OFF_HIGH_MAX):
            continue

        # EVバンド
        ev_r = expected_r_from_in_rank(in_rank, rr)
        if not (EV_MIN <= ev_r <= EV_MAX):
            continue

        # TPタッチ実績（現実性チェック）
        tp_hits = _tp_hit_count(hist, tp_price=tp_price, lookback=TP_HIT_LOOKBACK)
        if tp_hits < TP_HIT_MIN:
            continue

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                in_rank=in_rank,
                rr=float(rr),
                ev_r=float(ev_r),
                entry=float(entry),
                price_now=float(price_now),
                gap_pct=float(gap_pct),
                tp_price=float(tp_price),
                sl_price=float(sl_price),
                tp_hits=int(tp_hits),
            )
        )

    # Sort: EV -> RR -> TP hits -> smaller GAP
    cands.sort(key=lambda x: (x["ev_r"], x["rr"], x["tp_hits"], -x["gap_pct"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# レバ（ベース維持：地合いに応じたレンジ）
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.3, "強気（押し目が良いなら攻める）"
    if mkt_score >= 50:
        return 2.0, "中立（押し目メイン）"
    return 1.7, "弱め（新規は厳選）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# レポート
# ============================================================
def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    lev, _lev_comment = recommend_leverage(mkt_score)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events = build_event_warnings(today_date)
    swing = run_swing(today_date, mkt_score)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍")
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

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / TP実績あり）")
    if swing:
        rr_vals = [c["rr"] for c in swing]
        ev_vals = [c["ev_r"] for c in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f} / 平均EV:{float(np.mean(ev_vals)):.2f}")
        lines.append("")
        for i, c in enumerate(swing, start=1):
            star = " ⭐" if i == 1 else ""
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] {star}")
            lines.append(f"  RR:{c['rr']:.2f} EV:{c['ev_r']:.2f} IN:{c['in_rank']} TP実績:{c['tp_hits']}回/直近{TP_HIT_LOOKBACK}日")
            lines.append(f"  IN:{c['entry']:.1f} 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            lines.append(f"  TP:{c['tp_price']:.1f} SL:{c['sl_price']:.1f}")
            lines.append("")
    else:
        lines.append("- 該当なし（順張り＋押し目＋追いかけNG＋TP実績の条件に合う銘柄が無い）")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（前に届いた仕様：json={"text": ...} を維持）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL not set. Printing report only.")
        print(text)
        return

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
