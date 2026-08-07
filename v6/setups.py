"""v6 Layer 4 — セットアップ判定(VCP / 押し目 / ピボットブレイク)。

優先順位は VCP > 押し目 > ピボット。1銘柄につき1セットアップのみ割り当てる。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .filters import sma, atr_wilder

SETUP_VCP, SETUP_PULL, SETUP_PIVOT = "VCP", "押し目", "ピボット"


# ============ スイング検出 ============

def find_swings(daily: pd.DataFrame, lookback: int, min_bars: int) -> list:
    """フラクタル法でスイング高値・安値を検出する。

    min_bars本の左右より高い(安い)バーを転換点とする。戻り値は時系列順の
    [{"idx": 位置, "price": 価格, "kind": "H"/"L", "date": 日付}, ...]。

    min_bars=3 なら「左右3本ずつより高い」= 計7本の中心が高値。値が大きいほど
    検出されるスイングは少なく、大きな波だけが残る。
    """
    d = daily.iloc[-lookback:] if len(daily) > lookback else daily
    h, l = d["High"].values, d["Low"].values
    n = len(d)
    out = []
    for i in range(min_bars, n - min_bars):
        win_h = h[i - min_bars:i + min_bars + 1]
        win_l = l[i - min_bars:i + min_bars + 1]
        if h[i] == win_h.max() and (win_h.argmax() == min_bars):
            out.append({"idx": i, "price": float(h[i]), "kind": "H", "date": d.index[i]})
        elif l[i] == win_l.min() and (win_l.argmin() == min_bars):
            out.append({"idx": i, "price": float(l[i]), "kind": "L", "date": d.index[i]})
    return out


def _alternate(swings: list) -> list:
    """H/Lが交互になるよう整理する(連続する同種は極値のみ残す)。"""
    if not swings:
        return []
    out = [swings[0]]
    for s in swings[1:]:
        if s["kind"] == out[-1]["kind"]:
            # 同種が続いた場合、より極端な方を残す
            if (s["kind"] == "H" and s["price"] > out[-1]["price"]) or \
               (s["kind"] == "L" and s["price"] < out[-1]["price"]):
                out[-1] = s
        else:
            out.append(s)
    return out


# ============ 4-A. VCP(ミネルヴィニ・最優先) ============

def detect_vcp(daily: pd.DataFrame, cfg, diag: dict | None = None) -> dict | None:
    """収縮の連鎖(Volatility Contraction Pattern)を検出する。

    値幅の収縮だけでなく、出来高が枯れていること(供給が尽きたこと)を必須とする。
    仕様書§6-A: 出来高条件が最重要で、値幅収縮だけで通すと精度が大きく落ちる。

    diag: 与えられると、どの条件で落ちたかを集計する(本番でVCPが0件だった原因を
    特定するため。2026-08-07追加)。
    """
    def _no(reason):
        if diag is not None:
            diag[reason] = diag.get(reason, 0) + 1
        return None

    if len(daily) < cfg.vcp_lookback + 10:
        return _no("履歴不足")
    sw = _alternate(find_swings(daily, cfg.vcp_lookback, cfg.vcp_swing_min_bars))
    if len(sw) < 3:
        return _no("スイング3個未満")

    # ★2026-08-07修正: ベースの起点を特定し、そこ以降のスイングだけで収縮を測る。
    # 従来は130日窓の全スイングを対象にしていたため、ベース形成前の値動きや
    # ノイズまで収縮列に含まれ(検証では平均12.3回)、「全ての収縮が単調減少」を
    # 満たすことが事実上不可能だった(本番で119銘柄が全てこの条件で脱落)。
    # VCPは本来「ベースの中での収縮の連鎖」を指すため、ベース高値以降に限定するのが
    # 手法の定義に忠実。
    highs_all = [x for x in sw if x["kind"] == "H"]
    if not highs_all:
        return _no("高値なし")
    base_high = max(highs_all, key=lambda x: x["price"])
    sw_base = [x for x in sw if x["idx"] >= base_high["idx"]]
    if len(sw_base) < 3:
        return _no("ベース内のスイングが3個未満")

    # 高値→安値のペアから収縮率を作る(ベース内のみ)
    contractions = []
    marks = []
    for i in range(len(sw_base) - 1):
        a, b = sw_base[i], sw_base[i + 1]
        if a["kind"] == "H" and b["kind"] == "L" and a["price"] > 0:
            depth = (a["price"] - b["price"]) / a["price"]
            if depth > 0:
                contractions.append(depth)
                marks.append((a, b))
    if len(contractions) < cfg.vcp_min_contractions:
        return _no("収縮が2回未満")

    # 各回が前回の shrink_ratio 以下(=段階的に収縮している)
    shrinking = all(contractions[i + 1] < contractions[i] * cfg.vcp_shrink_ratio
                    for i in range(len(contractions) - 1))
    if not shrinking:
        return _no("段階的に収縮していない")
    if contractions[-1] > cfg.vcp_final_depth_max:
        return _no(f"最終収縮が{cfg.vcp_final_depth_max*100:.1f}%超")

    # 出来高の枯渇(最重要条件)
    v = daily["Volume"].dropna()
    if len(v) < 50:
        return _no("出来高履歴不足")
    vol_ma5 = float(v.iloc[-5:].mean())
    vol_ma50 = float(v.iloc[-50:].mean())
    if vol_ma50 <= 0 or vol_ma5 >= vol_ma50 * cfg.vcp_vol_dry_ratio:
        return _no("出来高が枯れていない")

    # 日柄(ロット): ベース形成日数
    base_start_idx = marks[0][0]["idx"]
    d = daily.iloc[-cfg.vcp_lookback:] if len(daily) > cfg.vcp_lookback else daily
    base_len = len(d) - base_start_idx
    if not (cfg.vcp_base_min_days <= base_len <= cfg.vcp_base_max_days):
        return _no("ベース日数が範囲外")

    # ピボット価格 = ベース内の直近スイング高値
    highs = [s["price"] for s in sw_base if s["kind"] == "H"]
    if not highs:
        return _no("高値なし")
    pivot = float(max(highs))        # ベース上限

    return {
        "setup": SETUP_VCP,
        "pivot": pivot,
        "contractions": [round(x, 4) for x in contractions],
        "n_contractions": len(contractions),
        "final_depth": round(contractions[-1], 4),
        "vol_ratio": round(vol_ma5 / vol_ma50, 3),
        "base_days": base_len,
        "swing_low": float(min(s["price"] for s in sw_base if s["kind"] == "L")),
        "recent_swing_low": float([s["price"] for s in sw_base if s["kind"] == "L"][-1]),
    }


# ============ 4-B. 押し目 ============

def detect_pullback(daily: pd.DataFrame, stage_d: int, cfg) -> dict | None:
    """上昇トレンド中の浅い押しを検出する。

    仕様書§6-B: 25日を超える押しは押し目ではなくベース形成であり、VCP側で扱う。
    出来高を伴って下げているのは押し目ではなく分配(distribution)なので除外する。
    """
    if stage_d != 1:
        return None
    c = daily["Close"].dropna()
    if len(c) < 60:
        return None

    # 直近30日の高値とその位置
    look = min(30, len(c))
    recent = c.iloc[-look:]
    recent_high = float(recent.max())
    hi_pos = int(np.argmax(recent.values))
    days_since_high = look - 1 - hi_pos
    close_now = float(c.iloc[-1])

    if recent_high <= 0:
        return None
    depth = (recent_high - close_now) / recent_high
    if not (cfg.pull_depth_min <= depth <= cfg.pull_depth_max):
        return None
    if not (cfg.pull_days_min <= days_since_high <= cfg.pull_days_max):
        return None

    atr = atr_wilder(daily["High"], daily["Low"], daily["Close"], cfg.atr_period)
    if pd.isna(atr.iloc[-1]) or float(atr.iloc[-1]) <= 0:
        return None
    atr_now = float(atr.iloc[-1])
    if (recent_high - close_now) > cfg.pull_atr_max * atr_now:
        return None

    ma = sma(c, cfg.pull_ma_days)
    if pd.isna(ma.iloc[-1]):
        return None
    ma_now = float(ma.iloc[-1])
    if close_now < ma_now * cfg.pull_ma_tolerance:
        return None

    # 押しの局面で出来高が減少していること(分配との識別)
    v = daily["Volume"].dropna()
    if len(v) < 20:
        return None
    if float(v.iloc[-3:].mean()) >= float(v.iloc[-20:].mean()):
        return None

    # 押し安値(高値以降の最安値)
    dip_low = float(daily["Low"].iloc[-(days_since_high + 1):].min()) if days_since_high >= 1 \
        else float(daily["Low"].iloc[-1])

    return {
        "setup": SETUP_PULL,
        "recent_high": recent_high,
        "depth": round(depth, 4),
        "days_since_high": days_since_high,
        "sma_pull": ma_now,
        "atr": atr_now,
        "recent_swing_low": dip_low,
        "vol_ratio": round(float(v.iloc[-3:].mean()) / float(v.iloc[-20:].mean()), 3),
    }


# ============ 4-C. ピボットブレイクアウト ============

def detect_pivot_breakout(daily: pd.DataFrame, cfg) -> dict | None:
    """ベース上限の上抜けを検出する。追いかけ禁止(3%以内)がLoK原則の実装。"""
    if len(daily) < 60:
        return None
    sw = _alternate(find_swings(daily, cfg.vcp_lookback, cfg.vcp_swing_min_bars))
    highs = [s for s in sw if s["kind"] == "H"]
    lows = [s for s in sw if s["kind"] == "L"]
    if not highs or not lows:
        return None

    # ピボット = 直近の確定スイング高値(当日を含まない)
    pivot = float(highs[-1]["price"])
    c_ok = daily["Close"].dropna()
    if len(c_ok) == 0:
        return None
    close_now = float(c_ok.iloc[-1])
    # ★NaN/異常値のガード。NaNは比較が常にFalseになり以降のガードを素通りするため、
    # 数値として妥当であることを明示的に確認してから先へ進める(2026-08-07)。
    if not (math.isfinite(close_now) and math.isfinite(pivot) and pivot > 0):
        return None
    if close_now <= pivot:
        return None

    ext = (close_now - pivot) / pivot
    if not math.isfinite(ext) or ext > cfg.pivot_max_extension:
        return None   # 追いかけない

    # ※ベースの深さによる足切りは行わない(仕様書§6-Cに無い条件のため)。
    # 深いベースのブレイクはストップ幅が8%を超え、§8-1の規定により
    # build_risk_plan で棄却される。棄却されること自体が意図された動作であり、
    # 検出側で先回りして絞るのは仕様外の介入になる。
    base_low = float(lows[-1]["price"])
    base_depth = (pivot - base_low) / pivot if (pivot > 0 and base_low > 0) else None

    v = daily["Volume"].dropna()
    if len(v) < 50:
        return None
    if float(v.iloc[-1]) < float(v.iloc[-50:].mean()) * cfg.pivot_vol_mult:
        return None

    hi, lo = float(daily["High"].iloc[-1]), float(daily["Low"].iloc[-1])
    if not (math.isfinite(hi) and math.isfinite(lo)) or hi <= lo:
        return None
    if close_now < (hi - lo) * cfg.pivot_close_position + lo:
        return None   # 当日の終値が高値圏にない

    return {
        "setup": SETUP_PIVOT,
        "pivot": pivot,
        "extension": round(ext, 4),
        "base_depth": (round(base_depth, 4) if base_depth is not None else None),
        "vol_ratio": round(float(v.iloc[-1]) / float(v.iloc[-50:].mean()), 2),
        "recent_swing_low": float(lows[-1]["price"]),
    }


# ============ 統合 ============

def detect_setup(daily: pd.DataFrame, stage_d: int, cfg, diag: dict | None = None) -> dict | None:
    """優先順位 VCP > 押し目 > ピボット で1つだけ返す。"""
    r = detect_vcp(daily, cfg, diag)
    if r:
        return r
    r = detect_pullback(daily, stage_d, cfg)
    if r:
        return r
    return detect_pivot_breakout(daily, cfg)
