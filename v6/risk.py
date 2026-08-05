"""v6 Layer 6 — リスク管理(エルダー2%/6% + ミネルヴィニ8%上限 + LoKのDD縮小)。

仕様書§12: このレイヤーは選定ロジック(Layer 4)より先に完成させること。
選定が未完成でもリスク管理があれば致命傷にはならないが、逆は成立しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskPlan:
    entry: float
    stop: float
    stop_pct: float          # エントリーからの距離(%)
    stop_kind: str           # "構造" or "8%上限"
    shares: int
    risk_amount: float       # 実際のリスク金額(円)
    risk_pct: float          # 口座に対する比率
    position_value: float
    target_2r: float
    rejected: str | None = None   # 棄却理由(あれば)


def build_risk_plan(entry: float, recent_swing_low: float, atr: float,
                    equity: float, cfg, risk_pct_override: float | None = None) -> RiskPlan:
    """ストップ価格・株数・リスク金額を一括算出する。

    ストップは「構造」と「8%上限」の狭い方を採る。構造ストップが8%より深い位置に
    ある銘柄は、そもそもリスクリワードが成立していないため棄却する(仕様書§8-1)。
    ストップを緩めて無理に取りにいかない、というのがミネルヴィニの原則。
    """
    structural = recent_swing_low - cfg.stop_atr_mult * atr
    hard = entry * cfg.hard_stop_pct
    stop = max(structural, hard)          # 価格が高い方 = 狭い方
    stop_kind = "構造" if structural >= hard else "8%上限"

    if stop <= 0 or stop >= entry:
        return RiskPlan(entry, stop, 0, stop_kind, 0, 0, 0, 0, 0,
                        rejected="ストップがエントリー以上")

    stop_distance = entry - stop
    stop_pct = stop_distance / entry

    # ★構造ストップが8%上限より深い = リスクリワードが成立しない → 棄却
    if structural < hard:
        return RiskPlan(entry, stop, stop_pct, stop_kind, 0, 0, 0, 0, 0,
                        rejected=f"構造ストップが-{(1-structural/entry)*100:.1f}%と"
                                 f"8%上限より深い(セットアップごと棄却)")

    risk_pct = risk_pct_override if risk_pct_override is not None else cfg.risk_per_trade
    risk_amount_target = equity * risk_pct
    raw = risk_amount_target / stop_distance
    shares = int(math.floor(raw / cfg.unit_shares) * cfg.unit_shares)

    # ポジション金額の上限キャップ
    max_value = equity * cfg.max_position_pct
    cap_shares = int(math.floor(max_value / entry / cfg.unit_shares) * cfg.unit_shares)
    shares = min(shares, cap_shares)

    if shares < cfg.unit_shares:
        return RiskPlan(entry, stop, stop_pct, stop_kind, 0, 0, 0, 0, 0,
                        rejected=f"1単元({cfg.unit_shares}株)に満たない"
                                 f"(必要資金{entry*cfg.unit_shares/1e4:.0f}万円)")

    risk_amount = stop_distance * shares
    return RiskPlan(
        entry=entry, stop=stop, stop_pct=stop_pct, stop_kind=stop_kind,
        shares=shares, risk_amount=risk_amount, risk_pct=risk_amount / equity,
        position_value=entry * shares,
        target_2r=entry + 2.0 * stop_distance,
    )


def portfolio_heat(positions: list, equity: float) -> float:
    """保有中の「現在値〜ストップまでの距離 × 株数」の合計を口座比で返す。

    エルダーの6%ルール: これが6%に達したら新規を止める。
    positions: [{"close":現在値, "stop":ストップ, "shares":株数}, ...]
    """
    if equity <= 0:
        return 0.0
    total = 0.0
    for p in positions:
        try:
            risk = (float(p["close"]) - float(p["stop"])) * float(p["shares"])
            total += max(0.0, risk)   # 既に含み益でストップが上なら0扱い
        except Exception:
            continue
    return total / equity


def drawdown_state(equity: float, peak_equity: float, cfg) -> dict:
    """ドローダウンに応じたリスク縮小・停止の判定(LoK: 大負け回避)。"""
    if peak_equity <= 0:
        return {"dd": 0.0, "risk_mult": 1.0, "halt": False, "note": ""}
    dd = max(0.0, (peak_equity - equity) / peak_equity)
    if dd >= cfg.dd_halt_th:
        return {"dd": dd, "risk_mult": 0.0, "halt": True,
                "note": f"DD {dd*100:.1f}% — 新規通知を完全停止(手動再開まで)"}
    if dd >= cfg.dd_reduce_th:
        return {"dd": dd, "risk_mult": 0.5, "halt": False,
                "note": f"DD {dd*100:.1f}% — 1トレードのリスクを半分に縮小"}
    return {"dd": dd, "risk_mult": 1.0, "halt": False, "note": ""}


def gate_new_entries(heat: float, dd: dict, cfg) -> tuple[bool, str]:
    """新規通知を出してよいか。(可否, 理由)を返す。"""
    if dd.get("halt"):
        return False, dd["note"]
    if heat >= cfg.portfolio_heat_max:
        return False, (f"ポートフォリオヒート {heat*100:.1f}% が上限"
                       f"{cfg.portfolio_heat_max*100:.0f}%に到達 — 新規停止"
                       "(保有銘柄の管理シグナルのみ配信)")
    return True, ""
