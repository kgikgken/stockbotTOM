"""STEP1: 地合いスコア(1〜5)・VI・全体ロット指示 — v4.1ルーブリック準拠."""

from __future__ import annotations

from .config import Config


def compute_macro(raw: dict, cfg: Config) -> dict:
    score = 3
    parts = []

    # 米国主要指数の方向
    spx, ndx = raw.get("spx"), raw.get("ndx")
    if spx is not None and ndx is not None:
        if spx > 0 and ndx > 0:
            score += 1; parts.append(f"米指数↑(S&P{spx:+.1f}%/NQ{ndx:+.1f}%) +1")
        elif spx < 0 and ndx < 0:
            score -= 1; parts.append(f"米指数↓(S&P{spx:+.1f}%/NQ{ndx:+.1f}%) -1")
        else:
            parts.append(f"米指数まちまち(S&P{spx:+.1f}%/NQ{ndx:+.1f}%) 0")
    else:
        parts.append("米指数: 欠落(暫定)")

    # SOX(重み高: 独立項目)
    sox = raw.get("sox")
    sox_rebound = False
    if sox is not None:
        if sox > 1.0:
            score += 1; sox_rebound = True; parts.append(f"SOX{sox:+.1f}% +1(明確反発)")
        elif sox < -1.0:
            score -= 1; parts.append(f"SOX{sox:+.1f}% -1")
        else:
            parts.append(f"SOX{sox:+.1f}% 0")
    else:
        parts.append("SOX: 欠落(暫定)")

    # ドル円: 水準でなく変化の速度・行き過ぎ度
    uj1, uj5 = raw.get("usdjpy_1d"), raw.get("usdjpy_5d")
    if uj1 is not None and uj5 is not None:
        if abs(uj1) >= 1.2 or abs(uj5) >= 2.5:
            score -= 1; parts.append(f"ドル円 急変(1d{uj1:+.1f}%/5d{uj5:+.1f}%) -1")
        else:
            parts.append(f"ドル円 緩やか(1d{uj1:+.1f}%/5d{uj5:+.1f}%) 0")
    else:
        parts.append("ドル円: 欠落(暫定)")

    # 日経先物
    nk = raw.get("nkfut")
    if nk is not None:
        if nk > 0.5:
            score += 1; parts.append(f"日経先物{nk:+.1f}% +1")
        elif nk < -0.5:
            score -= 1; parts.append(f"日経先物{nk:+.1f}% -1")
        else:
            parts.append(f"日経先物{nk:+.1f}% 0")
    else:
        parts.append("日経先物: 欠落(暫定)")

    # VI: 手動値優先、無ければ実現ボラproxy(その旨明記)
    vi = cfg.nikkei_vi_manual if cfg.nikkei_vi_manual > 0 else None
    vi_label = "日経VI(手動入力)"
    if vi is None:
        vi = raw.get("n225_rv20")
        vi_label = "N225実現ボラ20d(VI代理・単一ソース)"
    if vi is not None:
        if vi > cfg.vi_severe:
            score -= 2; parts.append(f"{vi_label}={vi:.1f} -2")
        elif vi > cfg.vi_warn:
            score -= 1; parts.append(f"{vi_label}={vi:.1f} -1")
        else:
            parts.append(f"{vi_label}={vi:.1f} 0")
    else:
        vi_label = "VI: 欠落(暫定)"
        parts.append(vi_label)

    vix = raw.get("vix")
    if vix is not None:
        parts.append(f"VIX={vix:.1f}(参考・採点外)")

    score = max(1, min(5, score))
    half = (vi is not None and vi > cfg.vi_half_lot) or score <= 2
    lot_factor = 0.5 if half else 1.0

    return {
        "score": score,
        "parts": parts,
        "vi": vi,
        "vi_label": vi_label,
        "half_lot": half,
        "lot_factor": lot_factor,
        "sox_rebound": sox_rebound,
        "provisional": raw.get("provisional", []),
        "fetched_at": raw.get("fetched_at", ""),
        "lot_text": ("通常の半分以下(VI>%.0f または スコア≤2)" % cfg.vi_half_lot) if half else "通常ロット可",
        "risk_cap": cfg.total_risk_cap_half if half else cfg.total_risk_cap,
    }
