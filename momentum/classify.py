"""STEP3: 3状態の日次分類 + エントリー水準の算出.

状態A(すでに流入): trend_align かつ 10/20日線付近の押し目 → 新規ロング候補
状態B(初動): VCP収縮 かつ 出来高を伴うドンチアン・ブレイク → 新規ロング候補
状態C(流出): 50日線割れ → 新規対象外(保有ポジションの手仕舞い判定にのみ使用)

ロングオンリー(空売りはしない)。モメンタムクラッシュの非対称リスクを踏まえ、
ショート側は仕様に含めない(研究レポートのロングオンリー推奨に準拠)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .indicators import chandelier_exit_long  # noqa: F401 (re-export for report use)


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    market: str
    state: str            # "A" or "B"
    score: float
    feat: dict
    entry: float = 0.0
    stop: float = 0.0
    chandelier: float = 0.0
    risk_w: float = 0.0
    risk_pct: float = 0.0
    shares: int = 0
    flags: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)


def _round_tick(p: float) -> float:
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def classify_state(feat: dict, cfg) -> str | None:
    """A / B / C / None(いずれでもない)を返す。優先順位: C(流出) > B(初動) > A(継続)。"""
    if feat["breakdown"]:
        return "C"
    if feat["vcp_now"] and feat["close"] > feat["donchian_prev"] \
            and feat["vol_ratio_today"] >= cfg.vcp_breakout_vol_mult:
        return "B"
    tol = cfg.pullback_tolerance_pct / 100
    near_fast = abs(feat["close"] / feat["sma10"] - 1) <= tol
    near_slow = abs(feat["close"] / feat["sma20"] - 1) <= tol
    if feat["trend_align"] and (near_fast or near_slow):
        return "A"
    return None


def build_candidate(row: dict, feat: dict, state: str, score: float, cfg) -> Candidate | None:
    """状態A/Bのみ新規ロング候補化。エントリー・初期ストップ・シャンデリア水準を算出。
    score はプール構築時にz-score母集団全体で算出済みの総合スコア(compute_pool_scores)を渡す。"""
    if state not in ("A", "B"):
        return None
    tkr = row["ticker"]
    code = tkr.replace(".T", "")
    entry = feat["close"]
    stop_raw = entry - cfg.initial_stop_atr_mult * feat["atr"]
    if stop_raw <= 0:
        return None
    risk_w = entry - stop_raw
    if risk_w <= 0 or risk_w / entry * 100 > cfg.max_risk_width_pct:
        return None

    c = Candidate(ticker=tkr, code=code, name=row.get("name", ""), sector=row.get("sector", ""),
                 market=row.get("market", "Prime"), state=state, score=score, feat=feat)
    c.entry = _round_tick(entry)
    c.stop = _round_tick(stop_raw)
    c.chandelier = _round_tick(feat["chandelier"]) if feat["chandelier"] == feat["chandelier"] else c.stop
    c.risk_w = risk_w

    if state == "A":
        c.flags.append("押し目買い: 10/20日線付近での反発を確認してからの発注を推奨")
    else:
        c.flags.append("ブレイク当日: 寄り付き直後の値動きで再確認してから発注(ダマシ注意)")
    c.checks = [
        f"出来高が平均比{feat['vol_ratio_today']:.1f}倍の裏取り(板・歩み値で確認)",
        f"ADX({feat['adx']:.0f}) — トレンド強度の参考値(閾値{cfg.adx_trend_th:.0f}は目安・厳格なゲートではない)",
        "同業種で他に強い銘柄が無いか(セクター全体の動きか個別かの確認)",
        "公開買付(TOB)・M&A等のコーポレートアクションが出ていないか(適時開示で確認)",
    ]
    return c
