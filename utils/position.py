from __future__ import annotations

from typing import List, Optional, Tuple

import re

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import SetupInfo, build_position_info, detect_setup, structure_sl_tp
from utils.rr_ev import calc_ev
from utils.util import atr14, adv20, atr_pct_last, safe_float


def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_ticker(t: str) -> str:
    """Normalize tickers for yfinance.

    Your project uses JP tickers (e.g. 6951.T). In positions.csv, users sometimes
    write them as `6951T`, `285AT`, or even `285A T`.

    This function:
      - removes spaces,
      - keeps tickers that already contain a dot as-is,
      - converts `XXXXT` / `XXXAT` -> `XXXX.T` / `XXXA.T` when it looks like a JP code.
    """
    s = str(t or "").strip().replace(" ", "")
    if not s:
        return ""
    if "." in s:
        return s
    # JP code patterns we want to fix: 3-5 digits + optional letter + trailing 'T'
    if re.match(r"^\d{3,5}[A-Z]?T$", s):
        return s[:-1] + ".T"
    # If user omitted the trailing 'T' but it's obviously a JP code, still append .T
    if re.match(r"^\d{3,5}[A-Z]?$", s):
        return s + ".T"
    return s


def _download_positions_history(tickers: List[str]) -> dict:
    """Best-effort bulk history fetch for positions.

    We keep this resilient:
      - If bulk download fails, fall back to per-ticker history().
      - We use auto_adjust=True to remain consistent with the rest of this project.
    """
    tickers = [str(t).strip() for t in (tickers or []) if str(t).strip()]
    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers=tickers,
            period="260d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    out: dict = {}

    # Single ticker: normal OHLC columns
    if isinstance(data, pd.DataFrame) and not isinstance(getattr(data, "columns", None), pd.MultiIndex):
        if not data.empty and {"Open", "High", "Low", "Close"}.issubset(set(data.columns)):
            out[tickers[0]] = data.dropna(how="all")
        return out

    # MultiIndex columns: (ticker, field)
    if isinstance(getattr(data, "columns", None), pd.MultiIndex):
        try:
            lvl0 = list(data.columns.levels[0])
        except Exception:
            lvl0 = []
        for t in tickers:
            if t not in lvl0:
                continue
            try:
                df_t = data[t].copy()
                if isinstance(df_t, pd.DataFrame) and not df_t.empty and "Close" in df_t.columns:
                    out[t] = df_t.dropna(how="all")
            except Exception:
                continue
    return out


def analyze_positions(
    df: pd.DataFrame,
    mkt_score: int,
    macro_on: bool,
    new_tickers: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """Analyze current positions and return (text, asset_estimate).

    Output is designed to be:
      - actionable (entry/current, SL/TP, unrealized R, next action),
      - consistent with the candidate section (RR/p/EV_R where possible),
      - resilient to missing columns (entry_price/quantity/setup).
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    new_set = set([
        _normalize_ticker(x)
        for x in (new_tickers or [])
        if str(x).strip()
    ])

    # Bulk fetch for speed; per-ticker fallback exists.
    tickers = []
    try:
        raw = (
            df.get("ticker", pd.Series(dtype=str)).astype(str).tolist()
            if df is not None and len(df)
            else []
        )
        tickers = [_normalize_ticker(t) for t in raw]
    except Exception:
        tickers = []

    hist_map = _download_positions_history(tickers)

    lines: List[str] = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = _normalize_ticker(row.get("ticker", ""))
        if not ticker:
            continue

        entry_price = safe_float(row.get("entry_price", np.nan), np.nan)
        qty = safe_float(row.get("quantity", np.nan), np.nan)

        hist = hist_map.get(ticker)
        if hist is None:
            try:
                hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            except Exception:
                hist = None

        cur = np.nan
        if hist is not None and isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
            try:
                cur = float(hist["Close"].iloc[-1])
            except Exception:
                cur = np.nan

        # Valuation (optional)
        value = np.nan
        if np.isfinite(cur) and np.isfinite(qty) and qty > 0:
            value = float(cur * qty)
            if np.isfinite(value) and value > 0:
                total_value += value

        # Build audit info (SL/TP + EV metrics) when entry_price exists.
        rr = np.nan
        cagr = np.nan
        p_hit = np.nan
        exp_days = np.nan
        sl = np.nan
        tp1 = np.nan
        tp2 = np.nan
        setup_used = ""

        info: Optional[SetupInfo] = None
        if hist is not None and isinstance(hist, pd.DataFrame) and len(hist) >= 60 and np.isfinite(entry_price) and entry_price > 0:
            # Primary audited path.
            info = build_position_info(hist, entry_price=float(entry_price), macro_on=bool(macro_on))

            # Fallback: if something goes wrong, still compute SL/TP from a setup hint.
            if info is None:
                try:
                    setup_hint = str(row.get("setup", "")).strip() or detect_setup(hist)[0]
                except Exception:
                    setup_hint = "A1"
                if setup_hint not in ("A1-Strong", "A1", "A2", "B", "POS"):
                    setup_hint = "A1"

                a = atr14(hist)
                atr_last = float(a.iloc[-1]) if a is not None and len(a) else np.nan
                if np.isfinite(atr_last) and atr_last > 0:
                    _sl, _tp1, _tp2, _rr_tp2, _expected_days = structure_sl_tp(
                        hist, float(entry_price), float(atr_last), bool(macro_on), setup_hint
                    )
                    risk = max(1e-6, float(entry_price) - float(_sl))
                    rr_tp1 = float((float(_tp1) - float(entry_price)) / risk)
                    info = SetupInfo(
                        setup=setup_hint,
                        tier=9,
                        entry_low=float(entry_price),
                        entry_high=float(entry_price),
                        sl=float(_sl),
                        tp1=float(_tp1),
                        tp2=float(_tp2),
                        rr=float(rr_tp1),
                        expected_days=float(_expected_days),
                        rday=float(rr_tp1 / max(0.5, float(_expected_days))),
                        trend_strength=1.0,
                        pullback_quality=1.0,
                        gu=False,
                        breakout_line=None,
                        adv20=float(adv20(hist)),
                        atrp=float(atr_pct_last(hist)),
                        entry_price=float(entry_price),
                        rr_tp1=float(rr_tp1),
                    )

        if info is not None:
            setup_used = str(info.setup)
            sl = safe_float(info.sl, np.nan)
            tp1 = safe_float(info.tp1, np.nan)
            tp2 = safe_float(info.tp2, np.nan)

            ev = calc_ev(info, mkt_score=mkt_score, macro_on=macro_on)
            rr = safe_float(ev.rr, np.nan)
            cagr = safe_float(ev.cagr_score, np.nan)
            p_hit = safe_float(ev.p_reach, np.nan)
            exp_days = safe_float(ev.expected_days, np.nan)

        # PnL
        pnl_pct = np.nan
        if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(cur) and cur > 0:
            pnl_pct = (cur - entry_price) / entry_price * 100.0

        # R-multiple (unrealized)
        r_now = np.nan
        if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(cur) and cur > 0 and np.isfinite(sl):
            risk = float(entry_price - sl)
            if np.isfinite(risk) and risk > 0:
                r_now = float((cur - entry_price) / risk)

        # Deterministic next action (telemetry; not a recommendation)
        action = "保有継続"
        if not (np.isfinite(entry_price) and entry_price > 0):
            action = "entry_price未設定（損益/R計算不可）"
        elif not (np.isfinite(cur) and cur > 0):
            action = "現在値取得失敗（要確認）"
        elif np.isfinite(sl) and cur <= sl:
            action = "SL到達/割れ（要対応）"
        elif np.isfinite(tp1) and cur >= tp1:
            action = "TP1到達（利確/SL引上げ検討）"
        elif np.isfinite(r_now) and r_now >= 1.0:
            action = "含み+1R（SL建値以上検討）"
        elif np.isfinite(r_now) and r_now >= 0.5:
            action = "含み+0.5R（SL引上げ検討）"
        elif np.isfinite(r_now) and r_now <= -0.5:
            action = "含み損（SL監視）"

        # Header
        lines.append(f"■ {ticker}")
        state_note = "保有中"
        if ticker in new_set:
            state_note += "（本日追加）"
        lines.append(f"・状態：{state_note}")

        # Prices
        if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(cur) and cur > 0:
            lines.append(f"・取得単価：{_fmt_yen(entry_price)} 円 / 現値：{_fmt_yen(cur)} 円")
        elif np.isfinite(entry_price) and entry_price > 0:
            lines.append(f"・取得単価：{_fmt_yen(entry_price)} 円 / 現値：-")
        elif np.isfinite(cur) and cur > 0:
            lines.append(f"・取得単価：- / 現値：{_fmt_yen(cur)} 円")
        else:
            lines.append("・取得単価：- / 現値：-")

        # Size/valuation (optional)
        if np.isfinite(qty) and qty > 0:
            if np.isfinite(value) and value > 0:
                lines.append(f"・数量：{qty:g} / 評価額：約{_fmt_yen(value)} 円")
            else:
                lines.append(f"・数量：{qty:g}")

        # PnL and R
        if np.isfinite(pnl_pct):
            if np.isfinite(r_now):
                lines.append(f"・損益：{pnl_pct:+.2f}%（含み {r_now:+.2f}R）")
            else:
                lines.append(f"・損益：{pnl_pct:+.2f}%")
        else:
            lines.append("・損益：—")

        # SL / TP distances
        if np.isfinite(cur) and cur > 0 and np.isfinite(sl) and sl > 0:
            sl_need = (sl / cur - 1.0) * 100.0  # negative if above SL
            lines.append(f"・想定SL：{_fmt_yen(sl)} 円（SLまで {sl_need:+.1f}%）")
        elif np.isfinite(sl) and sl > 0:
            lines.append(f"・想定SL：{_fmt_yen(sl)} 円")

        if np.isfinite(cur) and cur > 0 and np.isfinite(tp1) and tp1 > 0:
            tp_need = (tp1 / cur - 1.0) * 100.0
            lines.append(f"・想定TP1：{_fmt_yen(tp1)} 円（TP1まで {tp_need:+.1f}%）")
        elif np.isfinite(tp1) and tp1 > 0:
            lines.append(f"・想定TP1：{_fmt_yen(tp1)} 円")

        if np.isfinite(tp2) and tp2 > 0:
            lines.append(f"・想定TP2：{_fmt_yen(tp2)} 円")

        if setup_used:
            lines.append(f"・セットアップ：{setup_used}")

        lines.append(f"・次アクション：{action}")

        # Audit metrics (only if available)
        if np.isfinite(rr) and np.isfinite(p_hit):
            p_be = (1.0 / (rr + 1.0)) if rr > 0 else 1.0
            ev_r = (p_hit * rr) - ((1.0 - p_hit) * 1.0)
            lines.append(f"・到達確率（目安）：{p_hit:.3f}（損益分岐 p={p_be:.3f}）")
            lines.append(f"・期待値（R）：{ev_r:.2f}R")
        if np.isfinite(cagr):
            warn = "（要注意）" if cagr < 0.50 else ""
            lines.append(f"・CAGR寄与度（/日）：{cagr:.2f}{warn}")
        if np.isfinite(rr):
            lines.append(f"・RR（TP1基準）：{rr:.2f}")
        if np.isfinite(exp_days):
            lines.append(f"・想定日数（中央値）：{exp_days:.1f}日")

        lines.append("")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines).strip(), float(asset_est)
