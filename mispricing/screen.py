"""歪みハンティング本体 — v4.1 パス1(定量部)の忠実な自動化.

ボットの担当範囲(重要):
- ゲート1の定量トリガー検出(タイプE / A疑い・両方向)
- フィージビリティの機械的前段(流動性・ロット/ADV比・価格規制近接)
- R床・到達確率チェック・出口設計の円数字算出
- 確信度の素点合算(縮退-1は全銘柄に適用: 単一ソースのため)

ボットが「できない」こと(チャット側 = パス2の担当):
- ゲート0スティールマン / ゲート3反証・プレモータム(LLM判断)
- ニュース・開示の2ソース照合、タイプB/C/F/Gの検出
- 貸株在庫のバイナリ確認(iSPEEDでのユーザー確認が必要)
→ 従って本ボットの出力は全件「仮点灯(未確認)」であり、
  昇格にはiSPEED照合(独立2ソース化)+チャットでのゲート0/3が必要。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .config import Config
from .indicators import compute_features


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    market: str
    direction: str            # "ロング" / "ショート"
    mtype: str                # "E" / "A疑い"
    basis: str                # "正規化" / "縮退(バケット)"
    trigger_text: str
    feat: dict
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    ref2r: float = 0.0
    risk_w: float = 0.0
    plan_r: float = 2.0
    reach_days: float = 0.0
    conf: str = "中"
    conf_trail: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)   # iSPEED確認項目
    risk_pct: float = 0.0
    shares: int = 0
    rank_score: float = 0.0
    hold_days: int = 10
    expiry_days: int = 3
    trail_days: int = 3


CONF_ORDER = {"高": 3, "中": 2, "低": 1}


def _round_tick(p: float) -> float:
    """東証の呼値(TOPIX500外・通常銘柄の簡易版)。目安値のため簡易で足りる。"""
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def evaluate_stock(row: dict, df: pd.DataFrame, cfg: Config, macro: dict,
                   rejects: List[dict]) -> Candidate | None:
    tkr = row["ticker"]
    code = tkr.replace(".T", "")
    name, sector, market = row.get("name", ""), row.get("sector", ""), row.get("market", "Prime")

    def reject(stage: str, reason: str, close: float | None = None):
        rejects.append({"code": code, "name": name, "stage": stage, "reason": reason,
                        "close": round(close, 1) if close is not None else ""})

    feat = compute_features(df)
    if feat is None:
        return None  # データ不足はカウント外(棄却台帳を汚さない)

    close = feat["close"]

    # ---- フィージビリティ(機械的前段) ----
    if close < cfg.min_price:
        return None
    if feat["adv20_jpy"] < cfg.min_adv_jpy:
        return None
    if cfg.lot_jpy / max(feat["adv20_jpy"], 1.0) > cfg.max_lot_adv_ratio:
        reject("Feasibility", f"ロット/ADV比超過(ADV{feat['adv20_jpy']/1e8:.1f}億円)", close)
        return None

    # ---- ゲート1: 方向判定(正規化主基準 → バケット縮退) ----
    z, pctl, rsi, dev = feat["z_dev"], feat["pctl"], feat["rsi14"], feat["dev25"] * 100.0
    bdev = cfg.bucket_dev.get(market, cfg.bucket_dev["Standard"])
    brl = cfg.bucket_rsi_long.get(market, 25.0)
    brs = cfg.bucket_rsi_short.get(market, 75.0)

    direction = None
    basis = None
    if not math.isnan(z):
        if z <= -cfg.z_dev_th and pctl <= cfg.pctl_th and rsi < cfg.rsi_long:
            direction, basis = "ロング", "正規化"
        elif z >= cfg.z_dev_th and pctl >= (100 - cfg.pctl_th) and rsi > cfg.rsi_short:
            direction, basis = "ショート", "正規化"
    if direction is None:
        if dev <= -bdev and rsi < brl:
            direction, basis = "ロング", "縮退(バケット)"
        elif dev >= bdev and rsi > brs:
            direction, basis = "ショート", "縮退(バケット)"

    if direction is None:
        return None  # トリガー無し(棄却台帳の対象外: 「検討」に達していない)

    # ---- タイプ判定: A疑い(カタリスト痕跡) / E ----
    has_event = feat["event_signature"](cfg.event_lookback, cfg.event_ret_sigma, cfg.event_vol_z)
    mtype = "A疑い" if has_event else "E"

    trig = (f"25日乖離{dev:+.1f}% / z={z:+.2f} / pct={pctl:.0f}% / RSI14={rsi:.0f}"
            f" / 出来高z={feat['vol_z']:+.1f} [{basis}]")

    c = Candidate(ticker=tkr, code=code, name=name, sector=sector, market=market,
                  direction=direction, mtype=mtype, basis=basis, trigger_text=trig,
                  feat=feat, hold_days=cfg.hold_days, expiry_days=cfg.expiry_days,
                  trail_days=cfg.trail_days)

    # ---- 確信度: 素点合算(v4.1スタッキング規則) ----
    conf_pts = 3  # A/E上限=「高」から開始
    c.conf_trail.append("タイプ上限 高(A/E)")
    conf_pts -= 1
    c.conf_trail.append("単一ソース算出(yfinance)につき縮退扱い -1")
    if basis == "縮退(バケット)":
        conf_pts -= 1
        c.conf_trail.append("正規化不成立→バケット閾値 -1")

    # ゲート0の機械的注意フラグ(スティールマンの代替ではない)
    if direction == "ロング" and feat["down_trend"]:
        conf_pts -= 1
        c.conf_trail.append("200日線下向き&割れ(バリュートラップ域) -1")
        c.flags.append("長期下落トレンド内 — ゲート0で正当下落の可能性を要精査")
    if direction == "ショート" and feat["up_trend"]:
        conf_pts -= 1
        c.conf_trail.append("200日線上向き&上(順行トレンド) -1")
        c.flags.append("長期上昇トレンド内 — ゲート0で正当上昇の可能性を要精査")

    # 高ボラ規則(VI>閾値のとき)
    stop_mult = cfg.stop_atr_mult
    if macro["vi"] is not None and macro["vi"] > cfg.vi_half_lot and feat["atr_pct"] > cfg.hivol_atr_pct:
        if not macro["sox_rebound"]:
            reject("Feasibility", f"高ボラ×VI>{cfg.vi_half_lot:.0f}×SOX反発無し", close)
            return None
        conf_pts -= 1
        c.conf_trail.append("高ボラタグ(VI高・SOX反発で条件付き採用) -1")
        c.flags.append("高ボラ: 広め損切り・小ロット")
        stop_mult = cfg.stop_atr_mult_hivol

    c.conf = "高" if conf_pts >= 3 else ("中" if conf_pts == 2 else "低")

    # ---- ショートの追加フラグ ----
    if direction == "ショート":
        c.checks.append("貸株在庫/空売り可否(バイナリ特則: iSPEED画面確認で足りる)")
        c.checks.append("信用倍率・信用売り残(踏み上げリスク)")
        c.flags.append("ショート: 損失は理論上無限大 → ロングより保守的ロット必須")
        if feat["reg_10pct_hit"]:
            c.flags.append("空売り価格規制の発動可能性(基準値段比-10%接触) — 直近公表価格以下の新規売り不可/51単元以上適用/分割発注潜脱は違法")

    # ---- R床・出口設計(前日終値ベースの目安 → 寄り後調整必須) ----
    atr = feat["atr14"]
    if direction == "ロング":
        stop_raw = min(feat["low5"], close - stop_mult * atr) * (1 - cfg.stop_buffer_pct / 100)
        entry = close
        risk = entry - stop_raw
        room = feat["sma25"] - entry           # 平均回帰の構造的目標
    else:
        stop_raw = max(feat["high5"], close + stop_mult * atr) * (1 + cfg.stop_buffer_pct / 100)
        entry = close
        risk = stop_raw - entry
        room = entry - feat["sma25"]

    if risk <= 0:
        reject("R床", "リスク幅算出不能", close)
        return None
    if risk / entry * 100 > cfg.max_risk_width_pct:
        reject("R床", f"リスク幅{risk/entry*100:.1f}%>上限{cfg.max_risk_width_pct:.0f}%", close)
        return None
    if room < 2 * risk:
        reject("R床", f"値幅余地不足(余地{room/entry*100:+.1f}% < 2R={2*risk/entry*100:.1f}%)", close)
        return None

    reach_days = (2 * risk) / max(atr, 1e-9)
    if reach_days > cfg.hold_days:
        reject("到達確率", f"2R到達に約{reach_days:.0f}ATR日>{cfg.hold_days}日", close)
        return None

    sign = 1 if direction == "ロング" else -1
    c.entry = _round_tick(entry)
    c.stop = _round_tick(stop_raw)
    c.tp1 = _round_tick(entry + sign * risk)
    c.ref2r = _round_tick(entry + sign * 2 * risk)
    c.risk_w = risk
    c.reach_days = reach_days

    # ---- 確信度→リスク%(固定フラクショナル×地合い係数) ----
    if c.conf == "低":
        reject("確信度", "確信度低で不採用(累積減点) → 次点へ", close)
        c.risk_pct = 0.0
        c.rank_score = -1  # 次点行きマーカー(screen側で分岐)
    else:
        base = cfg.risk_pct_high if c.conf == "高" else cfg.risk_pct_mid
        c.risk_pct = base * macro["lot_factor"]
        if cfg.account_equity > 0:
            risk_amount = cfg.account_equity * c.risk_pct / 100
            c.shares = int(risk_amount / risk // 100 * 100)

    # ---- iSPEED確認リスト(独立2ソース化のための項目) ----
    c.checks = [
        f"RSI(14) 表示値 ≈ {rsi:.0f} か",
        f"25日線乖離率 ≈ {dev:+.1f}% か",
        "出来高急増の有無(カタリスト有無 → A/E確定)",
        "板の厚み(想定ロット消化可能か)",
    ] + c.checks

    # ---- ランクスコア ----
    zz = 0.0 if math.isnan(z) else min(abs(z), 4.0)
    rsix = max(0.0, (cfg.rsi_long - rsi)) if direction == "ロング" else max(0.0, rsi - cfg.rsi_short)
    c.rank_score = (zz + rsix / 10.0 + min(max(feat["vol_z"], 0.0), 3.0) * 0.3
                    + (0.5 if basis == "正規化" else 0.0)
                    + (0.2 if c.mtype == "A疑い" else 0.0)) if c.rank_score >= 0 else -1
    return c


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               cfg: Config, macro: dict) -> dict:
    rejects: List[dict] = []
    fired: List[Candidate] = []
    low_conf: List[Candidate] = []
    considered = 0

    for _, row in universe.iterrows():
        tkr = str(row["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None:
            continue
        c = evaluate_stock(row.to_dict(), df, cfg, macro, rejects)
        if c is None:
            continue
        considered += 1
        if c.rank_score < 0:
            low_conf.append(c)
        else:
            fired.append(c)

    fired.sort(key=lambda x: (-CONF_ORDER[x.conf], -x.rank_score))

    # ---- STEP4: 同時総オープンリスク上限(既存ポジ含む注記は report 側) ----
    cap = macro["risk_cap"]
    picked: List[Candidate] = []
    overflow: List[Candidate] = []
    total_risk = 0.0
    for c in fired:
        if len(picked) < cfg.max_candidates and total_risk + c.risk_pct <= cap + 1e-9:
            picked.append(c)
            total_risk += c.risk_pct
        else:
            reason = "総リスク上限超過" if len(picked) < cfg.max_candidates else "候補数上限"
            rejects.append({"code": c.code, "name": c.name, "stage": "STEP4",
                            "reason": f"{reason} → 次点", "close": round(c.feat["close"], 1)})
            overflow.append(c)

    runners = (overflow + low_conf)[: cfg.max_runners_up]

    # 検討n(トリガー到達=検討開始とみなす) / 棄却m / 採用k
    n_considered = considered + sum(1 for r in rejects if r["stage"] in ("Feasibility", "R床", "到達確率"))
    stats = {
        "considered": n_considered,
        "rejected": len(rejects),
        "picked": len(picked),
        "total_risk": total_risk,
        "risk_cap": cap,
        "by_stage": pd.Series([r["stage"] for r in rejects]).value_counts().to_dict() if rejects else {},
    }
    return {"picked": picked, "runners": runners, "rejects": rejects, "stats": stats}
