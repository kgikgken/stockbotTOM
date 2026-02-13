from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import SetupInfo, build_position_info, detect_setup, structure_sl_tp
from utils.rr_ev import calc_ev
from utils.util import atr14, adv20, atr_pct_last, safe_float

def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score: int, macro_on: bool) -> Tuple[str, float]:
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = safe_float(row.get("entry_price", 0), 0.0)
        qty = safe_float(row.get("quantity", 0), 0.0)

        cur = entry_price
        hist = None
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and not hist.empty:
                cur = float(hist["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        rr = np.nan
        cagr = np.nan
        p_hit = np.nan
        adj = np.nan
        exp_days = np.nan

        info: SetupInfo | None = None
        if hist is not None and len(hist) >= 60 and entry_price > 0:
            # First try the normal audited path (uses setup detection).
            if len(hist) >= 120:
                info = build_position_info(hist, entry_price=float(entry_price), macro_on=macro_on)

            # If setup detection fails (e.g., the position is already "out of pattern"),
            # fall back to a synthetic info object so the position remains auditable.
            if info is None or info.setup == "NONE":
                try:
                    setup_hint = str(row.get("setup", "")).strip() or detect_setup(hist)[0]
                except Exception:
                    setup_hint = "A1"
                if setup_hint not in ("A1-Strong", "A1", "A2", "B"):
                    setup_hint = "A1"

                a = atr14(hist)
                atr_last = float(a.iloc[-1]) if a is not None and len(a) else np.nan
                if np.isfinite(atr_last) and atr_last > 0:
                    sl, tp1, tp2, _rr_tp2, expected_days = structure_sl_tp(
                        hist, float(entry_price), float(atr_last), bool(macro_on), setup_hint
                    )
                    risk = max(1e-6, float(entry_price) - float(sl))
                    rr_tp1 = float((float(tp1) - float(entry_price)) / risk)
                    info = SetupInfo(
                        setup=setup_hint,
                        tier=1,
                        entry_low=float(entry_price),
                        entry_high=float(entry_price),
                        sl=float(sl),
                        tp1=float(tp1),
                        tp2=float(tp2),
                        rr=float(rr_tp1),
                        expected_days=float(expected_days),
                        rday=float(rr_tp1 / max(0.5, float(expected_days))),
                        trend_strength=1.0,
                        pullback_quality=1.0,
                        gu=False,
                        breakout_line=None,
                        adv20=float(adv20(hist)),
                        atrp=float(atr_pct_last(hist)),
                        entry_price=float(entry_price),
                        rr_tp1=float(rr_tp1),
                    )

        if info is not None and info.setup != "NONE":
            ev = calc_ev(info, mkt_score=mkt_score, macro_on=macro_on)
            rr = ev.rr
            adj = ev.adj_ev
            p_hit = ev.p_reach
            exp_days = ev.expected_days
            cagr = ev.cagr_score

        lines.append(f"■ {ticker}")
        lines.append("・状態：保有中（新規追加なし）")
        # Always show PnL as supplementary telemetry.
        lines.append(f"・損益：{pnl_pct:+.2f}%")

        if np.isfinite(rr) and np.isfinite(cagr):
            warn = "（要注意）" if cagr < 0.5 else ""
            lines.append(f"・CAGR寄与度（/日）：{cagr:.2f}{warn}")
            if np.isfinite(p_hit):
                lines.append(f"・到達確率（目安）：{p_hit:.2f}")
            if np.isfinite(adj):
                lines.append(f"・期待R×到達確率：{adj:.2f}")
            lines.append(f"・RR（TP1基準）：{rr:.2f}")
            if np.isfinite(exp_days):
                lines.append(f"・想定日数（中央値）：{exp_days:.1f}日")
        lines.append("")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines).strip(), float(asset_est)