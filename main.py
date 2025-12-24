from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import (
    detect_setup,
    gu_flag,
    in_zone_center,
    pwin_proxy,
    universe_pass,
)
from utils.rr import compute_tp_sl_rr  # positions用に残す（互換）
from utils.market import enhance_market_score


# ============================================================
# 設定（Swing専用 / 1〜7日）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外（営業日ではなく暦日：元仕様を維持）
EARNINGS_EXCLUDE_DAYS = 3

# セクター上位Nのみ
SECTOR_TOP_N = 5

# 出力上限
SWING_MAX_FINAL = 5
WATCH_MAX = 10

# 足切り
RR_MIN = 2.2
EV_MIN = 0.40
EXPECTED_DAYS_MAX = 5.0
R_PER_DAY_MIN = 0.50

# 追いかけ禁止（IN乖離）
IN_DIST_ATR_LIMIT = 0.80


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
        if not label:
            continue
        events.append(
            {
                "label": label,
                "kind": str(row.get("kind", "")).strip(),
                "date": str(row.get("date", "")).strip(),
                "time": str(row.get("time", "")).strip(),
                "datetime": str(row.get("datetime", "")).strip(),
            }
        )
    return events


def build_event_warnings(today_date) -> Tuple[List[str], bool]:
    """
    Returns: (lines, is_risky_day)
    is_risky_day: FOMC/CPI/日銀/GDP 等が「本日または翌日」なら True（保守的）
    """
    events = load_events()
    warns: List[str] = []
    risky = False

    keywords = ("FOMC", "CPI", "PCE", "雇用統計", "日銀", "GDP", "政策金利", "米CPI", "米PCE")

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

            if (delta in (0, 1)) and any(k in ev["label"] for k in keywords):
                risky = True

    if not warns:
        warns.append("- 特になし")
    return warns, risky


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

    keep: List[bool] = []
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
# MarketScore Δ3d（追加）
# ============================================================
def _market_score_from_prices(close: pd.Series) -> int:
    """
    close: 連続した終値（日足）
    スコア設計は「既存の軽量版」を踏襲：5日変化 + 0-100に丸め
    """
    if close is None or len(close) < 6:
        return 50
    c = close.astype(float).dropna()
    if len(c) < 6:
        return 50
    chg = (c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0
    base = 50.0 + float(np.clip(chg, -20.0, 20.0))
    return int(np.clip(round(base), 0, 100))


def calc_market_score_delta_3d() -> Tuple[int, int]:
    """
    Returns: (score_today, delta_3d)
    3営業日前を厳密に追わず、「直近3本前のバー終点」を近似に採用。
    """
    try:
        df = yf.Ticker("^TOPX").history(period="14d", auto_adjust=True)
        if df is None or df.empty or len(df) < 10:
            df = yf.Ticker("^N225").history(period="14d", auto_adjust=True)
        if df is None or df.empty:
            return 50, 0
        close = df["Close"].astype(float).dropna()
        if len(close) < 10:
            return 50, 0

        score_today = _market_score_from_prices(close)
        close_3d = close.iloc[:-3]  # 3本前まで
        score_3d_ago = _market_score_from_prices(close_3d)

        return int(score_today), int(score_today - score_3d_ago)
    except Exception:
        return 50, 0


# ============================================================
# レバレッジ（簡易）
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 60:
        return 2.0, "攻め（押し目＋一部ブレイク）"
    if mkt_score >= 45:
        return 1.7, "中立（厳選・押し目中心）"
    return 1.0, "守り（新規禁止）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# EV / AdjEV / 速度
# ============================================================
def calc_ev(pwin: float, rr: float) -> float:
    if rr <= 0:
        return -999.0
    p = float(np.clip(pwin, 0.0, 1.0))
    return float(p * rr - (1.0 - p) * 1.0)


def regime_multiplier(mkt_score: int, delta_3d: int, risky_event: bool) -> float:
    mult = 1.00
    if mkt_score >= 60 and delta_3d >= 0:
        mult *= 1.05
    if delta_3d <= -5:
        mult *= 0.70
    if risky_event:
        mult *= 0.75
    return float(mult)


def expected_days(tp2: float, entry: float, atr: float, k: float = 1.0) -> float:
    if not (np.isfinite(tp2) and np.isfinite(entry) and np.isfinite(atr) and atr > 0):
        return 999.0
    return float(max(0.5, (tp2 - entry) / (k * atr)))


# ============================================================
# スクリーニング
# ============================================================
@dataclass
class SwingCandidate:
    ticker: str
    name: str
    sector: str
    sector_rank: int
    setup: str  # A/B
    in_rank: str
    rr: float
    ev: float
    adjev: float
    r_per_day: float
    entry: float
    price_now: float
    gap_pct: float
    atr: float
    gu: bool
    stop: float
    tp1: float
    tp2: float
    exp_days: float
    action: str  # EXEC_NOW / LIMIT_WAIT / WATCH_ONLY


def run_swing(
    today_date,
    mkt_score: int,
    delta_mkt_3d: int,
    top_sectors: List[Tuple[str, float]],
    risky_event: bool,
) -> Tuple[List[SwingCandidate], List[SwingCandidate], List[str]]:
    """
    Returns: (main_list, watch_list, debug_reasons_for_notradecheck)
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], [], ["universe読み込み失敗"]

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], [], ["universeにticker/code列が無い"]

    # 決算除外（新規）
    uni = filter_earnings(uni, today_date)

    # セクター上位
    top_sector_names = [s for s, _ in top_sectors]
    sector_rank_map = {s: i + 1 for i, (s, _) in enumerate(top_sectors)}

    mains: List[SwingCandidate] = []
    watches: List[SwingCandidate] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # セクターフィルタ（上位のみ）
        if top_sector_names and (sector not in top_sector_names):
            continue

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        # Universe フィルタ
        ok, c_now, adv20, atr_pct = universe_pass(hist)
        if not ok:
            continue

        setup, in_rank = detect_setup(hist)
        if setup is None:
            continue

        # Setup B は地合い55未満は弾く（事故回避）
        if setup == "B" and mkt_score < 55:
            continue

        center, atr = in_zone_center(hist, setup)
        if not (np.isfinite(center) and np.isfinite(atr) and atr > 0):
            continue

        # INゾーン（中心のみ使う：±は報告用に不要なら後で追加）
        entry = float(center)

        # GU判定
        gu = gu_flag(hist)

        # 追いかけ禁止（IN乖離）
        price_now = float(hist["Close"].astype(float).iloc[-1])
        dist_atr = abs(price_now - entry) / atr if atr > 0 else 999.0

        # Stop/Target（仕様ベース）
        if setup == "A":
            stop = entry - 1.2 * atr
            breakline = np.nan
            resistance = float(hist["Close"].astype(float).tail(60).max()) * 0.995
        else:
            # Setup B: ブレイクラインは entry（=HH20prev）
            breakline = entry
            stop = breakline - 1.0 * atr
            resistance = float(hist["Close"].astype(float).tail(60).max()) * 0.995

        # R距離
        r_dist = entry - stop
        if not (np.isfinite(r_dist) and r_dist > 0):
            continue

        # TP2: 基本は 3R だが、60日高値手前で頭を叩く
        tp2_raw = entry + 3.0 * r_dist
        tp2 = min(tp2_raw, resistance)
        if tp2 <= entry:
            continue

        # TP1: 1.5R（TP2に合わせて上限）
        tp1 = min(entry + 1.5 * r_dist, tp2 * 0.995)

        # RR（TP2基準）
        rr = (tp2 - entry) / r_dist

        # 期待日数 / 速度
        exp_days = expected_days(tp2, entry, atr, k=1.0)
        r_day = rr / exp_days if exp_days > 0 else 0.0

        # 代理Pwin → EV
        srank = int(sector_rank_map.get(sector, 99))
        pwin = pwin_proxy(in_rank, setup, srank, mkt_score, delta_mkt_3d, adv20)
        ev = calc_ev(pwin, rr)
        mult = regime_multiplier(mkt_score, delta_mkt_3d, risky_event)
        adjev = ev * mult

        # 足切り
        if rr < RR_MIN:
            continue
        if ev < EV_MIN:
            continue
        if exp_days > EXPECTED_DAYS_MAX:
            continue
        if r_day < R_PER_DAY_MIN:
            continue

        # ギャップ%
        gap_pct = (price_now / entry - 1.0) * 100.0 if entry > 0 else float("nan")

        # Action
        action = "LIMIT_WAIT"
        if gu:
            action = "WATCH_ONLY"
        elif dist_atr > IN_DIST_ATR_LIMIT:
            action = "WATCH_ONLY"
        else:
            # 即INは「かなり厳選」（勝てる寄せ）
            # - Setup Aのみ
            # - 現値がINから近い（0.5ATR以内）
            # - 速度/AdjEVが上位
            if (setup == "A") and (dist_atr <= 0.50) and (adjev >= 0.80) and (r_day >= 0.80):
                action = "EXEC_NOW"
            else:
                action = "LIMIT_WAIT"

        cand = SwingCandidate(
            ticker=ticker,
            name=name,
            sector=sector,
            sector_rank=srank,
            setup=str(setup),
            in_rank=in_rank,
            rr=float(rr),
            ev=float(ev),
            adjev=float(adjev),
            r_per_day=float(r_day),
            entry=float(round(entry, 1)),
            price_now=float(round(price_now, 1)),
            gap_pct=float(gap_pct),
            atr=float(round(atr, 1)),
            gu=bool(gu),
            stop=float(round(stop, 1)),
            tp1=float(round(tp1, 1)),
            tp2=float(round(tp2, 1)),
            exp_days=float(round(exp_days, 1)),
            action=action,
        )

        if action == "WATCH_ONLY":
            watches.append(cand)
        else:
            mains.append(cand)

    # ソート：AdjEV * R/day を最優先（=最強に寄せる）
    def key(c: SwingCandidate):
        return (c.adjev * c.r_per_day, c.adjev, c.r_per_day, c.rr)

    mains.sort(key=key, reverse=True)
    watches.sort(key=key, reverse=True)

    # 本命は mains の先頭から（最大5）
    mains = mains[:SWING_MAX_FINAL]
    watches = watches[:WATCH_MAX]

    return mains, watches, []


# ============================================================
# NO-TRADE 判定
# ============================================================
def decide_trade_allowed(
    mkt_score: int,
    delta_3d: int,
    risky_event: bool,
    mains: List[SwingCandidate],
    watches: List[SwingCandidate],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if mkt_score < 45:
        reasons.append("MarketScore<45")
    if (delta_3d <= -5) and (mkt_score < 55):
        reasons.append("ΔMarketScore_3d<=-5 かつ MarketScore<55")

    # 上位候補（mains）の平均AdjEV
    if mains:
        avg_adjev = float(np.mean([c.adjev for c in mains]))
        if avg_adjev < 0.30:
            reasons.append("上位候補の平均AdjEV<0.30")
    else:
        # main が空なら「実質ノートレ」
        reasons.append("条件を満たす候補なし")

    # GU比率（mains+watches のうちGU）
    all_cands = mains + watches
    if all_cands:
        gu_ratio = float(np.mean([1.0 if c.gu else 0.0 for c in all_cands]))
        if gu_ratio >= 0.60:
            reasons.append("GU_flag比率>=60%")

    # イベント（強制ノートレにはしない、倍率で抑制するだけ。必要ならここで追加）
    if risky_event:
        # 仕様では倍率 0.75、NO-TRADEではない
        pass

    allowed = len(reasons) == 0
    return allowed, reasons


# ============================================================
# レポート（日本語のみ）
# ============================================================
def _yn(b: bool) -> str:
    return "Y" if b else "N"


def _action_jp(a: str) -> str:
    if a == "EXEC_NOW":
        return "即IN可"
    if a == "LIMIT_WAIT":
        return "指値待ち"
    return "監視のみ"


def build_report(
    today_str: str,
    today_date,
    mkt_score: int,
    mkt_comment: str,
    delta_3d: int,
    lev: float,
    lev_comment: str,
    max_pos: int,
    sectors: List[Tuple[str, float]],
    event_lines: List[str],
    trade_allowed: bool,
    notrade_reasons: List[str],
    mains: List[SwingCandidate],
    watches: List[SwingCandidate],
    pos_text: str,
) -> str:
    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用）")
    if trade_allowed:
        lines.append("✅ 新規可（条件クリア）")
    else:
        rs = " / ".join(notrade_reasons) if notrade_reasons else "条件該当"
        lines.append(f"🚫 本日は新規見送り（{rs}）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta_3d:+d}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors, 1):
            lines.append(f"{i}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(event_lines)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if mains:
        avg_rr = float(np.mean([c.rr for c in mains]))
        avg_ev = float(np.mean([c.ev for c in mains]))
        avg_adjev = float(np.mean([c.adjev for c in mains]))
        avg_rpd = float(np.mean([c.r_per_day for c in mains]))
        lines.append(f"  候補数:{len(mains)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均AdjEV:{avg_adjev:.2f} / 平均R/day:{avg_rpd:.2f}")
        lines.append("")

        # 本命（上位1〜2）
        core = mains[:2]
        lines.append("🎯 本命（1〜2銘柄）")
        for c in core:
            lines.append(f"- {c.ticker} {c.name} [{c.sector}] ⭐")
            lines.append(f"  Setup:{c.setup}  RR:{c.rr:.2f}  AdjEV:{c.adjev:.2f}  R/day:{c.r_per_day:.2f}")
            lines.append(f"  IN:{c.entry:.1f} 現在:{c.price_now:.1f} ({c.gap_pct:+.2f}%)  ATR:{c.atr:.1f}  GU:{_yn(c.gu)}")
            lines.append(f"  STOP:{c.stop:.1f}  TP1:{c.tp1:.1f}  TP2:{c.tp2:.1f}  ExpectedDays:{c.exp_days:.1f}  行動:{_action_jp(c.action)}")
            lines.append("")

        # 残り
        rest = mains[2:]
        if rest:
            lines.append("👀 監視・指値")
            for c in rest:
                lines.append(f"- {c.ticker} {c.name} [{c.sector}]")
                lines.append(f"  Setup:{c.setup}  RR:{c.rr:.2f}  AdjEV:{c.adjev:.2f}  R/day:{c.r_per_day:.2f}")
                lines.append(f"  IN:{c.entry:.1f} 現在:{c.price_now:.1f} ({c.gap_pct:+.2f}%)  ATR:{c.atr:.1f}  GU:{_yn(c.gu)}")
                lines.append(f"  STOP:{c.stop:.1f}  TP1:{c.tp1:.1f}  TP2:{c.tp2:.1f}  ExpectedDays:{c.exp_days:.1f}  行動:{_action_jp(c.action)}")
                lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # WATCH_ONLY（任意表示）
    if watches:
        lines.append("👁 監視リスト（今日は入らない）")
        for c in watches[:WATCH_MAX]:
            lines.append(f"- {c.ticker} {c.name} [{c.sector}]")
            lines.append(f"  Setup:{c.setup}  RR:{c.rr:.2f}  AdjEV:{c.adjev:.2f}  R/day:{c.r_per_day:.2f}  GU:{_yn(c.gu)}  行動:{_action_jp(c.action)}")
        lines.append("")

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

    # 既存の市場評価（表示用コメントなど）
    mkt = enhance_market_score()
    mkt_comment = str(mkt.get("comment", "中立"))

    # Δ3d（必須追加）
    mkt_score_today, delta_3d = calc_market_score_delta_3d()
    mkt_score = int(mkt_score_today)

    # レバ
    lev, lev_comment = recommend_leverage(mkt_score)
    max_pos = 0

    # セクター
    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)

    # イベント
    event_lines, risky_event = build_event_warnings(today_date)

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0
    max_pos = calc_max_position(total_asset, lev)

    # 候補
    mains, watches, _ = run_swing(
        today_date=today_date,
        mkt_score=mkt_score,
        delta_mkt_3d=delta_3d,
        top_sectors=sectors,
        risky_event=risky_event,
    )

    # NO-TRADE
    trade_allowed, reasons = decide_trade_allowed(
        mkt_score=mkt_score,
        delta_3d=delta_3d,
        risky_event=risky_event,
        mains=mains,
        watches=watches,
    )
    if not trade_allowed:
        # 新規禁止の日はレバ表示も守りに寄せる（実行はしない）
        lev = 1.0
        lev_comment = "新規禁止"
        max_pos = calc_max_position(total_asset, lev)

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt_score=mkt_score,
        mkt_comment=mkt_comment,
        delta_3d=delta_3d,
        lev=lev,
        lev_comment=lev_comment,
        max_pos=max_pos,
        sectors=sectors,
        event_lines=event_lines,
        trade_allowed=trade_allowed,
        notrade_reasons=reasons,
        mains=mains,
        watches=watches,
        pos_text=pos_text,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
