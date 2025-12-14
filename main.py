from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# stockbotTOM v11.1 PERFECT (spec-complete)
# - NO-TRADE DAY 強制
# - 理論RRだけ高い銘柄の排除（構造TP / 到達率）
# - Swing×Day 競合制御（同一銘柄/同一セクター）
# - 寄り前 + 寄り後再判定（RUN_MODE）
# ============================================================


# ============================================================
# 設定（環境変数で上書き可）
# ============================================================
UNIVERSE_PATH = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")
POSITIONS_PATH = os.getenv("POSITIONS_PATH", "positions.csv")
EVENTS_PATH = os.getenv("EVENTS_PATH", "events.csv")
WORKER_URL = os.getenv("WORKER_URL")

# 実行モード: "preopen"（寄り前） / "postopen"（寄り後）
RUN_MODE = os.getenv("RUN_MODE", "preopen").strip().lower()

# 取得データ
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "260"))  # 到達率計算の都合で多め

# スクリーニング抽出
SCREENING_TOP_N = int(os.getenv("SCREENING_TOP_N", "25"))
MAX_FINAL_SWING = int(os.getenv("MAX_FINAL_SWING", "3"))
MAX_FINAL_DAY = int(os.getenv("MAX_FINAL_DAY", "5"))

# 決算フィルタ
EARNINGS_EXCLUDE_DAYS = int(os.getenv("EARNINGS_EXCLUDE_DAYS", "3"))

# 流動性フィルタ（売買代金）
LIQ_MIN_TURNOVER = float(os.getenv("LIQ_MIN_TURNOVER", "100000000"))  # 1億円/日

# NO-TRADE DAY（v11.1確定）
NO_TRADE_MKT_SCORE_TH = float(os.getenv("NO_TRADE_MKT_SCORE_TH", "45"))
NO_TRADE_SWING_AVG_EV_TH = float(os.getenv("NO_TRADE_SWING_AVG_EV_TH", "0.3"))

# Day GU危険域（寄り前: trigger距離 / 寄り後: 実ギャップ）
MAX_GU_DANGER_ATR_PREOPEN = float(os.getenv("MAX_GU_DANGER_ATR_PREOPEN", "0.8"))
MAX_GU_DANGER_ATR_POSTOPEN = float(os.getenv("MAX_GU_DANGER_ATR_POSTOPEN", "1.2"))

# 理論RR排除（追加補助: ATRベース）
MIN_STOP_ATR = float(os.getenv("MIN_STOP_ATR", "0.7"))
MIN_TARGET_ATR = float(os.getenv("MIN_TARGET_ATR", "1.0"))

# v11.1確定：構造TPチェック（20日高値×1.02）
TP_OVER_20D_HIGH_MULT = float(os.getenv("TP_OVER_20D_HIGH_MULT", "1.02"))
TP_OVER_PENALTY_MULT = float(os.getenv("TP_OVER_PENALTY_MULT", "0.7"))
TP_OVER_EXCLUDE_MULT = float(os.getenv("TP_OVER_EXCLUDE_MULT", "1.06"))  # ここ超えたら除外

# v11.1確定：構造的到達率 < 60% → EV×0.7
REACH_RATE_LOOKBACK_EVENTS = int(os.getenv("REACH_RATE_LOOKBACK_EVENTS", "30"))  # 直近イベント数
REACH_RATE_FORWARD_DAYS = int(os.getenv("REACH_RATE_FORWARD_DAYS", "10"))        # 先読み日数
REACH_RATE_TH = float(os.getenv("REACH_RATE_TH", "0.60"))
REACH_RATE_EV_MULT = float(os.getenv("REACH_RATE_EV_MULT", "0.7"))

# 競合制御
SECTOR_SWING_LIMIT_IF_DAY_PRESENT = int(os.getenv("SECTOR_SWING_LIMIT_IF_DAY_PRESENT", "1"))


# ============================================================
# JST
# ============================================================
JST = timezone(timedelta(hours=9))

def jst_now() -> datetime:
    return datetime.now(tz=JST)

def jst_today_date() -> datetime.date:
    return jst_now().date()

def jst_today_str() -> str:
    return jst_today_date().strftime("%Y-%m-%d")


# ============================================================
# 既存 utils があれば利用（無ければフォールバック）
# ============================================================
def _try_import():
    out = {}
    try:
        from utils.market import enhance_market_score  # type: ignore
        out["enhance_market_score"] = enhance_market_score
    except Exception:
        out["enhance_market_score"] = None

    try:
        from utils.market import calc_market_score  # type: ignore
        out["calc_market_score"] = calc_market_score
    except Exception:
        out["calc_market_score"] = None

    try:
        from utils.sector import top_sectors_5d  # type: ignore
        out["top_sectors_5d"] = top_sectors_5d
    except Exception:
        out["top_sectors_5d"] = None

    try:
        from utils.position import load_positions, analyze_positions  # type: ignore
        out["load_positions"] = load_positions
        out["analyze_positions"] = analyze_positions
    except Exception:
        out["load_positions"] = None
        out["analyze_positions"] = None

    try:
        from utils.scoring import score_stock  # type: ignore
        out["score_stock"] = score_stock
    except Exception:
        from utils_scoring_v11_1_perfect import score_stock  # type: ignore
        out["score_stock"] = score_stock

    return out

U = _try_import()


# ============================================================
# CSV loaders
# ============================================================
def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("universe_jpx.csv に ticker 列が必要です")
    # sector/name は無くても動く
    return df

def load_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "event", "importance"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["date", "event", "importance"])

def parse_date_safe(x) -> Optional[datetime.date]:
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None

def is_earnings_excluded(earnings_date: Optional[datetime.date], today: datetime.date, n: int) -> bool:
    if earnings_date is None:
        return False
    return abs((earnings_date - today).days) <= n


# ============================================================
# Market/sector/positions
# ============================================================
def compute_market_score() -> float:
    if U.get("enhance_market_score"):
        try:
            return float(U["enhance_market_score"]())
        except Exception:
            pass
    if U.get("calc_market_score"):
        try:
            return float(U["calc_market_score"]())
        except Exception:
            pass
    return 50.0

def get_top_sectors_5d() -> List[Tuple[str, float]]:
    fn = U.get("top_sectors_5d")
    if not fn:
        return []
    try:
        out = fn()
        return out if isinstance(out, list) else []
    except Exception:
        return []

def get_positions_summary() -> str:
    load_positions = U.get("load_positions")
    analyze_positions = U.get("analyze_positions")
    if not load_positions or not analyze_positions:
        return "(positions.csv 未使用)"
    try:
        pos_df = load_positions(POSITIONS_PATH)
        analysis = analyze_positions(pos_df)
        lev = float(analysis.get("recommended_leverage", analysis.get("leverage", 1.0)))
        max_pos = float(analysis.get("max_position_yen", analysis.get("max_position", 0.0)))
        return f"レバ: {lev:.1f}倍 / MAX建玉: {max_pos:,.0f}円"
    except Exception:
        return "(positions.csv 解析失敗)"

def build_event_warnings(events_df: pd.DataFrame, today: datetime.date, window_days: int = 2) -> List[str]:
    if events_df is None or events_df.empty or "date" not in events_df.columns:
        return []
    out: List[str] = []
    for _, r in events_df.iterrows():
        d = parse_date_safe(r.get("date"))
        if d is None:
            continue
        delta = (d - today).days
        if -1 <= delta <= window_days:
            name = str(r.get("event", "")).strip() or "イベント"
            out.append(f"{name}（{d.strftime('%Y-%m-%d')}）")
    return out[:6]


# ============================================================
# yfinance
# ============================================================
def yf_symbol_jp(ticker: str) -> str:
    t = str(ticker).strip()
    if t.endswith(".T") or t.endswith(".OS") or t.endswith(".JP"):
        return t
    return f"{t}.T"

def fetch_ohlcv(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    period = f"{max(lookback_days * 2, 365)}d"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(lookback_days).reset_index(drop=True)
    # 欠損行を除去
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


# ============================================================
# NO-TRADE DAY
# ============================================================
@dataclass
class NoTradeDecision:
    no_trade: bool
    reasons: List[str]

def decide_no_trade_day(mkt_score: float, swing_avg_ev_r: float, day_all_gu_danger: bool) -> NoTradeDecision:
    reasons: List[str] = []
    if mkt_score < NO_TRADE_MKT_SCORE_TH:
        reasons.append(f"mkt_score < {NO_TRADE_MKT_SCORE_TH:.0f}")
    if swing_avg_ev_r < NO_TRADE_SWING_AVG_EV_TH:
        reasons.append(f"Swing候補の平均EV < {NO_TRADE_SWING_AVG_EV_TH:.1f}R")
    if day_all_gu_danger:
        reasons.append("Day候補が全て「GU危険域」")
    return NoTradeDecision(no_trade=len(reasons) > 0, reasons=reasons)


# ============================================================
# LINE
# ============================================================
def post_to_worker(message: str) -> None:
    if not WORKER_URL:
        print("WORKER_URL 未設定。出力のみ。")
        print(message)
        return
    r = requests.post(WORKER_URL, json={"message": message}, timeout=25)
    print(f"Worker response: {r.status_code} {r.text[:200]}")


# ============================================================
# 競合制御（v11.1確定）
# ============================================================
def resolve_conflicts(day_df: pd.DataFrame, swing_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    warnings: List[str] = []

    # 1) 同一銘柄：併用禁止。高スコア側だけ残す
    if len(day_df) and len(swing_df):
        day_tickers = set(day_df["ticker"].astype(str))
        swing_tickers = set(swing_df["ticker"].astype(str))
        common = day_tickers & swing_tickers
        if common:
            keep_day = []
            keep_swing = []
            for t in common:
                drow = day_df[day_df["ticker"] == t].iloc[0]
                srow = swing_df[swing_df["ticker"] == t].iloc[0]
                if float(drow.get("total_score", 0)) >= float(srow.get("total_score", 0)):
                    keep_day.append(t)
                else:
                    keep_swing.append(t)

            day_df = pd.concat([day_df[~day_df["ticker"].isin(list(common))], day_df[day_df["ticker"].isin(keep_day)]], ignore_index=True)
            swing_df = pd.concat([swing_df[~swing_df["ticker"].isin(list(common))], swing_df[swing_df["ticker"].isin(keep_swing)]], ignore_index=True)

    # 2) 同一セクター：DayがあるセクターはSwingを最大1銘柄
    if len(day_df) and len(swing_df) and "sector" in day_df.columns and "sector" in swing_df.columns:
        day_sectors = set(day_df["sector"].astype(str))
        restricted = []
        kept_rows = []
        for sec in day_sectors:
            cand = swing_df[swing_df["sector"].astype(str) == sec].sort_values(["ev_r_adj", "total_score"], ascending=False)
            if len(cand) > SECTOR_SWING_LIMIT_IF_DAY_PRESENT:
                restricted.append(sec)
                kept = cand.head(SECTOR_SWING_LIMIT_IF_DAY_PRESENT)
                kept_rows.append(kept)
                swing_df = pd.concat([swing_df[swing_df["sector"].astype(str) != sec], kept], ignore_index=True)

        if restricted:
            warnings.append("⚠ 同セクターDayあり → Swingは1銘柄制限")

    # 整列
    if len(day_df):
        day_df = day_df.sort_values(["total_score"], ascending=False).reset_index(drop=True)
    if len(swing_df):
        swing_df = swing_df.sort_values(["ev_r_adj", "total_score"], ascending=False).reset_index(drop=True)

    return day_df, swing_df, warnings


# ============================================================
# スクリーニング
# ============================================================
def run_screening() -> str:
    today = jst_today_date()
    uni = load_universe(UNIVERSE_PATH)
    events = load_events(EVENTS_PATH)

    mkt_score = compute_market_score()
    sectors_5d = get_top_sectors_5d()
    pos_summary = get_positions_summary()

    scored: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        earnings_date = parse_date_safe(row.get("earnings_date", None))
        if is_earnings_excluded(earnings_date, today, EARNINGS_EXCLUDE_DAYS):
            continue

        symbol = yf_symbol_jp(ticker)
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 120:
            continue

        # 流動性（最新日の close*volume）
        turnover = float(df["close"].iloc[-1] * df["volume"].iloc[-1])
        if turnover < LIQ_MIN_TURNOVER:
            continue

        meta = row.to_dict()
        meta["sector"] = str(row.get("sector", "")).strip()
        meta["name"] = str(row.get("name", "")).strip()

        s = U["score_stock"](ticker=ticker, df=df, meta=meta, run_mode=RUN_MODE,
                            cfg=dict(
                                MIN_STOP_ATR=MIN_STOP_ATR,
                                MIN_TARGET_ATR=MIN_TARGET_ATR,
                                TP_OVER_20D_HIGH_MULT=TP_OVER_20D_HIGH_MULT,
                                TP_OVER_PENALTY_MULT=TP_OVER_PENALTY_MULT,
                                TP_OVER_EXCLUDE_MULT=TP_OVER_EXCLUDE_MULT,
                                REACH_RATE_LOOKBACK_EVENTS=REACH_RATE_LOOKBACK_EVENTS,
                                REACH_RATE_FORWARD_DAYS=REACH_RATE_FORWARD_DAYS,
                                REACH_RATE_TH=REACH_RATE_TH,
                                REACH_RATE_EV_MULT=REACH_RATE_EV_MULT,
                                MAX_GU_DANGER_ATR_PREOPEN=MAX_GU_DANGER_ATR_PREOPEN,
                                MAX_GU_DANGER_ATR_POSTOPEN=MAX_GU_DANGER_ATR_POSTOPEN,
                            ))

        if not isinstance(s, dict):
            continue

        s["ticker"] = ticker
        s["name"] = meta["name"]
        s["sector"] = meta["sector"]
        s["turnover"] = turnover
        scored.append(s)

    df_all = pd.DataFrame(scored) if scored else pd.DataFrame(columns=["ticker"])

    # 必須列の整備
    for c, default in [
        ("mode", "swing"),
        ("total_score", 0.0),
        ("rr_adj", 0.0),
        ("ev_r_adj", 0.0),
        ("gu_danger", False),
        ("tp_price", np.nan),
        ("sl_price", np.nan),
        ("in_price", np.nan),
        ("in_diff_pct", np.nan),
        ("reach_rate", np.nan),
        ("rr_raw", 0.0),
        ("ev_r_raw", 0.0),
        ("reject_reason", ""),
    ]:
        if c not in df_all.columns:
            df_all[c] = default

    # Top抽出（まずは総合スコア）
    top = df_all.sort_values("total_score", ascending=False).head(SCREENING_TOP_N).copy()

    swing = top[top["mode"] == "swing"].copy()
    day = top[top["mode"] == "day"].copy()

    # Final（スコア順で一旦切る）
    swing_final = swing.sort_values(["ev_r_adj", "total_score"], ascending=False).head(MAX_FINAL_SWING).copy()
    day_final = day.sort_values(["total_score"], ascending=False).head(MAX_FINAL_DAY).copy()

    # 競合制御（v11.1確定）
    day_final, swing_final, conflict_warn = resolve_conflicts(day_final, swing_final)

    # Swing平均EV
    swing_avg_ev = float(swing_final["ev_r_adj"].mean()) if len(swing_final) else 0.0
    # Day全GU危険域
    day_all_gu = bool(len(day_final) > 0 and day_final["gu_danger"].astype(bool).all())

    # NO-TRADE判定（v11.1確定）
    nt = decide_no_trade_day(mkt_score=mkt_score, swing_avg_ev_r=swing_avg_ev, day_all_gu_danger=day_all_gu)

    # LINEメッセージ
    lines: List[str] = []
    lines.append(f"📅 {jst_today_str()} stockbotTOM 日報 (v11.1)  [{RUN_MODE}]")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score:.0f}点")
    lines.append(f"- {pos_summary}")

    if nt.no_trade:
        lines.append("")
        lines.append("🚫 本日は新規見送り日（条件該当）")
        for r in nt.reasons:
            lines.append(f"- {r}")

    if sectors_5d:
        lines.append("")
        lines.append("📈 セクター（5日）")
        for i, (sec, pct) in enumerate(sectors_5d[:5], start=1):
            lines.append(f"{i}. {sec} ({pct:+.2f}%)")

    warn = build_event_warnings(events, today)
    if warn:
        lines.append("")
        lines.append("⚠ イベント")
        for w in warn:
            lines.append(f"- {w}")

    if conflict_warn:
        lines.append("")
        for w in conflict_warn:
            lines.append(w)

    def fmt_price(x) -> str:
        try:
            if pd.isna(x):
                return "-"
            return f"{float(x):.1f}"
        except Exception:
            return "-"

    def fmt_pct(x) -> str:
        try:
            if pd.isna(x):
                return "-"
            return f"{float(x):+.2f}%"
        except Exception:
            return "-"

    # Swing
    lines.append("")
    lines.append("📌 Swing（最大3）")
    if len(swing_final) == 0:
        lines.append("- 該当なし")
    else:
        for _, r in swing_final.iterrows():
            t = str(r.get("ticker", ""))
            name = str(r.get("name", "")).strip()
            sec = str(r.get("sector", "")).strip()
            in_p = fmt_price(r.get("in_price"))
            diff = fmt_pct(r.get("in_diff_pct"))
            tp = fmt_price(r.get("tp_price"))
            sl = fmt_price(r.get("sl_price"))
            rr = float(r.get("rr_adj", 0.0) or 0.0)
            ev = float(r.get("ev_r_adj", 0.0) or 0.0)
            reach = r.get("reach_rate", np.nan)
            reach_s = "-" if pd.isna(reach) else f"{float(reach)*100:.0f}%"

            lines.append(f"- {t} {name} [{sec}]")
            lines.append(f"  IN目安: {in_p}（現在との差 {diff}） / TP {tp} / SL {sl}")
            lines.append(f"  RR {rr:.2f} / EV {ev:.2f}R / 構造到達率 {reach_s}")

    if len(swing_final):
        lines.append(f"(Swing平均EV: {swing_avg_ev:.2f}R)")

    # Day
    lines.append("")
    lines.append("⚡ Day（成行前提）")
    if len(day_final) == 0:
        lines.append("- 該当なし")
    else:
        for _, r in day_final.iterrows():
            t = str(r.get("ticker", ""))
            name = str(r.get("name", "")).strip()
            sec = str(r.get("sector", "")).strip()
            tp = fmt_price(r.get("tp_price"))
            sl = fmt_price(r.get("sl_price"))
            rr = float(r.get("rr_adj", 0.0) or 0.0)
            ev = float(r.get("ev_r_adj", 0.0) or 0.0)
            gu = bool(r.get("gu_danger", False))
            note = " / GU危険域" if gu else ""
            lines.append(f"- {t} {name} [{sec}] / TP {tp} / SL {sl} / RR {rr:.2f} / EV {ev:.2f}R{note}")

        if day_all_gu:
            lines.append("(Day候補は全てGU危険域)")

    lines.append("")
    if nt.no_trade:
        lines.append("✅ ルール: 本日は新規エントリー禁止（監視のみ）")
    else:
        lines.append("✅ ルール: 条件一致のみエントリー（無理はしない）")

    return "\n".join(lines).strip() + "\n"


def main():
    msg = run_screening()
    post_to_worker(msg)


if __name__ == "__main__":
    main()
