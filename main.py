from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score, market_score_delta_3d
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


# ============================================================
# 設定（Swing専用 / ベース保持）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外（新規）
EARNINGS_EXCLUDE_DAYS = 3

# Swing条件
SWING_MAX_FINAL = 5
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40

# 1〜7日（速度重視）
MAX_EXPECTED_DAYS = 5.0
MIN_R_PER_DAY = 0.50

# GU / IN乖離
GU_ATR_TH = 1.0
IN_DIST_ATR_TH = 0.8

# NO-TRADE
NO_TRADE_MKT_SCORE = 45
NO_TRADE_DELTA3_TH = -5
NO_TRADE_DELTA3_SCORE_CAP = 55
NO_TRADE_AVG_ADJ_EV = 0.40
NO_TRADE_GU_RATIO = 0.60

# 表示
SECTOR_TOP_N = 5


# ============================================================
# util
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
                date=str(r.get("date", "")).strip(),
                time=str(r.get("time", "")).strip(),
                datetime=str(r.get("datetime", "")).strip(),
            )
        )
    return out


def build_event_warnings(today_date) -> Tuple[List[str], bool]:
    events = load_events()
    warns: List[str] = []
    near_critical = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days

        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

        # 前日〜当日は環境悪化扱い（v1.1）
        if 0 <= delta <= 1:
            near_critical = True

    if not warns:
        warns.append("- 特になし")
    return warns, near_critical


# ============================================================
# earnings
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    keep = []
    for x in d:
        if pd.isna(x):
            keep.append(True)
        else:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
    return df[keep]


# ============================================================
# EV
# ============================================================
def expected_r(in_rank: str, rr: float) -> float:
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return float(win * rr - (1.0 - win))


def regime_multiplier(mkt_score: int, delta3: int, event_near: bool) -> float:
    mult = 1.0
    if mkt_score >= 60 and delta3 >= 0:
        mult *= 1.05
    if delta3 <= NO_TRADE_DELTA3_TH:
        mult *= 0.70
    if event_near:
        mult *= 0.75
    return float(mult)


def _setup_type(in_rank: str) -> str:
    # v1.1：A=押し目（強） / B=ブレイク寄り（通常）
    if in_rank == "強IN":
        return "A"
    if in_rank == "通常IN":
        return "B"
    return "?"


def _action_type(price_now: float, entry: float, atr: float, gu_flag: bool) -> str:
    # 追いかけ禁止を機械化
    # - GUは問答無用で監視
    # - IN中心からの乖離(ATR単位)で EXEC / LIMIT / WATCH を決める
    if gu_flag:
        return "WATCH_ONLY"
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(price_now) and np.isfinite(entry) and entry > 0):
        return "WATCH_ONLY"

    dist = abs(price_now - entry) / atr

    # 0.8ATR超は「今日は入らない」
    if dist > IN_DIST_ATR_TH:
        return "WATCH_ONLY"

    # 0.4〜0.8ATRは指値待ち（押し目待ち）
    if dist > 0.4:
        return "LIMIT_WAIT"

    return "EXEC_NOW"



# ============================================================
# Swing screening（順張り専用）
# ============================================================
def run_swing(today_date, mkt_score: int, delta3: int, event_near: bool) -> Tuple[List[Dict], Dict]:
    """
    戻り:
      - final_candidates
      - stats: dict(avg_adj_ev, gu_ratio, no_trade, reasons, avg_rr, avg_ev, avg_rpd)
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], {"no_trade": True, "reasons": ["universe_read_fail"]}

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], {"no_trade": True, "reasons": ["ticker_column_missing"]}

    uni = filter_earnings(uni, today_date)

    mult = regime_multiplier(mkt_score, delta3, event_near)

    cands: List[Dict] = []
    gu_count = 0

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        # --- TrendGate（逆張り完全排除） ---
        if not trend_gate(hist):
            continue

        score = score_stock(hist)
        if score is None or not np.isfinite(score) or score < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < SWING_RR_MIN:
            continue

        ev = expected_r(in_rank, rr)
        if not np.isfinite(ev) or ev < SWING_EV_R_MIN:
            continue

        adj_ev = float(ev * mult)

        atr = float(rr_info.get("atr", 0.0))
        if not (np.isfinite(atr) and atr > 0):
            atr = max(float(hist["Close"].iloc[-1]) * 0.01, 1.0)

        # 速度（ExpectedDays, R/day）
        entry = float(rr_info["entry"])
        tp2 = float(rr_info["tp_price"])  # 既存のTP（TP2扱い）
        expected_days = (tp2 - entry) / (atr * 1.0) if atr > 0 else 999.0
        expected_days = float(expected_days) if np.isfinite(expected_days) else 999.0
        r_per_day = float(rr / expected_days) if expected_days > 0 else 0.0

        # 速度足切り（1〜7日戦）
        if expected_days > MAX_EXPECTED_DAYS:
            continue
        if r_per_day < MIN_R_PER_DAY:
            continue

        price_now = _safe_float(hist["Close"].iloc[-1])
        open_today = _safe_float(hist["Open"].iloc[-1])
        prev_close = _safe_float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price_now

        gu_flag = bool(np.isfinite(open_today) and np.isfinite(prev_close) and open_today > (prev_close + GU_ATR_TH * atr))
        if gu_flag:
            gu_count += 1

        action = _action_type(price_now, entry, atr, gu_flag)

        gap = (price_now / entry - 1) * 100 if entry > 0 else np.nan

        # TP1/TP2
        sl_price = float(rr_info["sl_price"])
        r_unit = max(entry - sl_price, 0.0)
        tp1 = entry + 1.5 * r_unit
        tp2 = float(rr_info["tp_price"])

        cands.append(
            dict(
                ticker=ticker,
                name=str(r.get("name", ticker)),
                sector=str(r.get("sector", r.get("industry_big", "不明"))),
                setup=_setup_type(in_rank),
                in_rank=in_rank,
                rr=float(rr),
                ev=float(ev),
                adj_ev=float(adj_ev),
                r_per_day=float(r_per_day),
                expected_days=float(expected_days),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap) if np.isfinite(gap) else np.nan,
                atr=float(atr),
                gu_flag=gu_flag,
                action=action,
                sl_price=float(sl_price),
                tp1=float(tp1),
                tp2=float(tp2),
            )
        )

    # 統計
    gu_ratio = (gu_count / len(cands)) if cands else 0.0
    avg_adj_ev = float(np.mean([x["adj_ev"] for x in cands])) if cands else 0.0
    avg_rr = float(np.mean([x["rr"] for x in cands])) if cands else 0.0
    avg_ev = float(np.mean([x["ev"] for x in cands])) if cands else 0.0
    avg_rpd = float(np.mean([x["r_per_day"] for x in cands])) if cands else 0.0

    reasons: List[str] = []

    # NO-TRADE判定（候補が0でも、地合いで止める）
    no_trade = False
    if mkt_score < NO_TRADE_MKT_SCORE:
        no_trade = True
        reasons.append("MarketScore<45")
    if (delta3 <= NO_TRADE_DELTA3_TH) and (mkt_score < NO_TRADE_DELTA3_SCORE_CAP):
        no_trade = True
        reasons.append("Δ3d悪化")
    if cands and avg_adj_ev < NO_TRADE_AVG_ADJ_EV:
        no_trade = True
        reasons.append("AvgAdjustedEV不足")
    if cands and gu_ratio >= NO_TRADE_GU_RATIO:
        no_trade = True
        reasons.append("GU過多")

    # ソート：AdjustedEV → R/day → RR
    cands.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    # セクター偏り（同一セクター最大2）
    picked: List[Dict] = []
    sec_cnt: Dict[str, int] = {}
    for c in cands:
        sec = str(c.get("sector", "不明"))
        if sec_cnt.get(sec, 0) >= 2:
            continue
        picked.append(c)
        sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if len(picked) >= SWING_MAX_FINAL:
            break

    # NO-TRADEなら、Actionを全部WATCH_ONLYに落とす（入らない日を固定）
    if no_trade:
        for c in picked:
            c["action"] = "WATCH_ONLY"

    stats = {
        "no_trade": no_trade,
        "reasons": reasons,
        "gu_ratio": float(gu_ratio),
        "avg_adj_ev": float(avg_adj_ev),
        "avg_rr": float(avg_rr),
        "avg_ev": float(avg_ev),
        "avg_rpd": float(avg_rpd),
        "mult": float(mult),
    }
    return picked, stats


# ============================================================
# レバレッジ（ベース保持：地合い依存は残すが、判断はNO-TRADEが握る）
# ============================================================
def recommend_leverage(mkt_score: int) -> float:
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 45:
        return 1.7
    return 0.0


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# レポート
# ============================================================
def build_report(today_str, today_date, mkt: Dict, delta3: int, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    events, event_near = build_event_warnings(today_date)

    lev = recommend_leverage(mkt_score)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(SECTOR_TOP_N)
    swing, stats = run_swing(today_date, mkt_score=mkt_score, delta3=delta3, event_near=event_near)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用）")

    if stats.get("no_trade", False):
        reason_str = " / ".join(stats.get("reasons", []) or ["条件該当"])
        lines.append(f"🚫 本日は新規見送り（{reason_str}）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3:+d}")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
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

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if swing:
        lines.append(
            f"  候補数:{len(swing)}銘柄 / 平均RR:{stats.get('avg_rr',0):.2f} / 平均EV:{stats.get('avg_ev',0):.2f} / 平均AdjEV:{stats.get('avg_adj_ev',0):.2f} / 平均R/day:{stats.get('avg_rpd',0):.2f}"
        )
        lines.append("")
        for c in swing:
            star = " ⭐" if c.get("action") == "EXEC_NOW" else ""
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] {star}")
            lines.append(
                f"  Setup:{c.get('setup','?')}  RR:{c['rr']:.2f}  EV:{c['ev']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_per_day']:.2f}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)  ATR:{c['atr']:.1f}  GU:{'Y' if c['gu_flag'] else 'N'}"
            )
            lines.append(
                f"  STOP:{c['sl_price']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  Action:{c['action']}"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")
    return "\n".join(lines)


# ============================================================
# LINE送信（通った仕様：json={"text": ...}）
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
    delta3 = market_score_delta_3d()

    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        delta3=delta3,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
