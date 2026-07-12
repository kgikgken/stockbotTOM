"""歪みハンティング本体 — v5.0 パス1(定量部)の自動化.

エンジン構成(v5.0):
- エンジンA(主力・ロング): 業種内リバーサル×非ファンダ性×クオリティ
    (1)非ファンダ性: ボットは「イベント痕跡」の有無で代理判定。痕跡ありは
       一段深いオーバーシュートを要求し、TDnetニュース確認をユーザー確認項目に積む。
    (2)クオリティ・キル: 財務データはボットから未確認 → キルは解除できない(保守側)。
       5項目のチェックリストを全候補に添付し、確認前の実弾エントリーを規律違反と明記。
    (3)リバーサル・トリガー: 業種指数比の相対乖離(正規化主基準 z≤-2 & 下位5%)、
       押し目日数2〜5営業日。縮退はバケット閾値(相対乖離%)→さらに業種系列が
       作れない場合のみ絶対乖離バケット。
- エンジンS(ショート・(b)逆流戻り売りのみ自動検出): 流出セクター(成熟以降)の
  2〜3日リバウンド。確信度上限「中」→単一ソース-1で常に「低」=参考層止まり。
  (a)悪材料イベント型は公式開示が根拠のためチャット側。
- エンジンB(PEAD疑い): 決算月(2/5/8/11)のみ。上方イベント痕跡+ドリフト限定的。
  「B疑い」として参考層/確認リストへ(決算上振れの確認はユーザー)。
- エンジンG: イベントカレンダー依存のためチャット側。
- 旧C/E: エンジンAの補助表示(RSI・絶対乖離)に統合。単独点灯根拠にしない。

ボット出力は全件「仮点灯(未確認・単一ソース)」。本命昇格はiSPEED照合＋
チャット側ゲート0/3(スティールマン・反証)後。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .indicators import compute_features
from .flowmap import flow_tag

CONF_ORDER = {"高": 3, "中": 2, "低": 1}
QUALITY_KILL_ITEMS = "営業赤字(直近四半期)/営業CF赤字(通期)/3ヶ月内下方修正/自己資本比率低(製30・非製20%)/監理整理・継続疑義"


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    market: str
    direction: str            # ロング / ショート
    engine: str               # A / S / B疑い
    basis: str                # 正規化(業種内) / 縮退(バケット) / 縮退(絶対乖離) / 定性
    nonfund: str              # 非ファンダ性の代理判定
    trigger_text: str
    feat: dict
    ftag: str = "中立"        # 順流/逆流/中立
    sec_stage: str = "-"
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    ref2r: float = 0.0
    risk_w: float = 0.0
    net2r: float = 2.0
    reach_days: float = 0.0
    conf: str = "中"
    conf_trail: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    risk_pct: float = 0.0
    shares: int = 0
    rank_score: float = 0.0
    hold_days: int = 5
    expiry_days: int = 3
    trail_days: int = 3
    watch_reason: str = ""    # 参考監視層に載る場合の理由


def _round_tick(p: float) -> float:
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def _rel5(df: pd.DataFrame, sec_log: pd.Series | None, w: int = 5):
    """業種指数比の5日相対リターン系列(対数差)とその z / percentile。"""
    if sec_log is None:
        return None, np.nan, np.nan, np.nan
    logc = np.log(df["Close"].dropna())
    logc, sl = logc.align(sec_log, join="inner")
    if len(logc) < 260:
        return None, np.nan, np.nan, np.nan
    rel = (logc.diff(w) - sl.diff(w)).dropna()
    win = rel.iloc[-252:]
    if len(win) < 200:
        return None, np.nan, np.nan, np.nan
    now = float(rel.iloc[-1])
    mu, sd = float(win.mean()), float(win.std(ddof=0))
    z = (now - mu) / sd if sd > 1e-12 else np.nan
    pctl = float((win <= now).mean() * 100.0)
    return rel, now, z, pctl


def _pead_suspect(df: pd.DataFrame, cfg: Config):
    """上方イベント痕跡(3〜N日前)+ドリフト限定的 → B疑い。(day_ago, event_ret, drift%) or None"""
    c = df["Close"].dropna()
    v = df["Volume"].reindex(c.index)
    if len(c) < 80:
        return None
    lr = np.log(c / c.shift(1))
    sd60 = float(lr.iloc[-61:-1].std(ddof=0))
    vm, vs = float(v.iloc[-21:-1].mean()), float(v.iloc[-21:-1].std(ddof=0))
    if sd60 <= 1e-9 or vs <= 1e-9:
        return None
    for k in range(3, min(cfg.pead_lookback, len(c) - 2) + 1):
        r = float(lr.iloc[-k])
        vz = (float(v.iloc[-k]) - vm) / vs
        if r >= 2.5 * sd60 and vz >= 2.0 and r > 0:
            ev_close = float(c.iloc[-k])
            drift = (float(c.iloc[-1]) / ev_close - 1.0) * 100.0
            if 0.0 <= drift < cfg.pead_max_drift_pct:
                return k, r * 100, drift
            return None
    return None


def evaluate_stock(row: dict, df: pd.DataFrame, cfg: Config, macro: dict,
                   flow: dict, month: int, rejects: List[dict]) -> Candidate | None:
    tkr = row["ticker"]
    code = tkr.replace(".T", "")
    name, sector, market = row.get("name", ""), row.get("sector", ""), row.get("market", "Prime")

    def reject(stage: str, reason: str, close=None):
        rejects.append({"code": code, "name": name, "stage": stage, "reason": reason,
                        "close": round(close, 1) if close is not None else ""})

    feat = compute_features(df)
    if feat is None:
        return None
    close = feat["close"]

    # ---- 流動性フィルタ(v5.0: ADV5億円・ロット/ADV比は自動充足) ----
    if close < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
        return None

    sec_ret = flow.get("sector_ret", {})
    sec_log = None
    if sector in sec_ret:
        sec_log = np.log1p(sec_ret[sector].fillna(0)).cumsum()
    _, rel_now, rel_z, rel_pctl = _rel5(df, sec_log, cfg.flow_window)
    rel_pct = rel_now * 100 if rel_now is not None and not (isinstance(rel_now, float) and math.isnan(rel_now)) else np.nan

    z25, rsi, dev = feat["z_dev"], feat["rsi14"], feat["dev25"] * 100.0
    engine = None
    direction = None
    basis = None
    nonfund = "-"

    # ================= エンジンA: 業種内リバーサル(ロング) =================
    a_trigger = False
    if not math.isnan(rel_z):
        if rel_z <= -cfg.rel_z_th and rel_pctl <= cfg.rel_pctl_th:
            a_trigger, basis = True, "正規化(業種内)"
        elif not math.isnan(rel_pct) and rel_pct <= -cfg.bucket_reldev.get(market, 10.0):
            a_trigger, basis = True, "縮退(バケット)"
    elif dev <= -cfg.bucket_dev.get(market, 15.0) and rsi < cfg.bucket_rsi_long.get(market, 25.0):
        a_trigger, basis = True, "縮退(絶対乖離)"  # 業種系列が作れない場合のみ

    if a_trigger:
        # 歯(3): 押し目日数 2〜5営業日
        if not (cfg.dip_min_days <= feat["dip_days"] <= cfg.dip_max_days):
            reject("エンジンA", f"押し目日数{feat['dip_days']}日(適合2〜{cfg.dip_max_days})", close)
            return None
        # 歯(1): 非ファンダ性の代理判定
        has_event = feat["event_signature"](max(feat["dip_days"], cfg.event_lookback),
                                            cfg.event_ret_sigma, cfg.event_vol_z)
        if has_event:
            deeper = False
            if basis == "正規化(業種内)" and rel_z <= -cfg.rel_z_th * cfg.event_overshoot_mult:
                deeper = True
            if basis == "縮退(バケット)" and not math.isnan(rel_pct) and \
                    rel_pct <= -cfg.bucket_reldev.get(market, 10.0) * cfg.event_overshoot_mult:
                deeper = True
            if basis == "縮退(絶対乖離)" and dev <= -cfg.bucket_dev.get(market, 15.0) * cfg.event_overshoot_mult:
                deeper = True
            if not deeper:
                reject("エンジンA", "イベント痕跡×一段深い乖離なし(CFニュース疑い・非ファンダ性未確認)", close)
                return None
            nonfund = "イベント痕跡あり→深押し要件クリア(TDnet要確認)"
        else:
            nonfund = "痕跡なし(純テクニカル・最上位確度/最終確認要)"
        engine, direction = "A", "ロング"

    # ================= エンジンS(b): 逆流戻り売り(ショート) =================
    if engine is None and flow.get("ok"):
        if sector in flow["outflow_set"] and flow["sector_stage"].get(sector, "") == "成熟以降":
            c = df["Close"].dropna()
            low10 = float(c.iloc[-11:].min())
            rebound_atr = (close - low10) / max(feat["atr14"], 1e-9)
            if (cfg.rebound_min_days <= feat["rebound_days"] <= cfg.rebound_max_days
                    and rebound_atr >= cfg.rebound_min_atr and close < feat["sma25"]):
                engine, direction, basis = "S", "ショート", "定性(逆流戻り売り)"
                nonfund = "流出セクター内リバウンドの反落狙い"

    # ================= エンジンB疑い: PEAD(決算月のみ) =================
    if engine is None and month in cfg.engine_b_months:
        p = _pead_suspect(df, cfg)
        if p:
            k, ev_ret, drift = p
            engine, direction, basis = "B疑い", "ロング", "定性(PEAD痕跡)"
            nonfund = f"{k}営業日前に+{ev_ret:.0f}%イベント/以降ドリフト+{drift:.1f}%(決算上振れ要確認)"

    if engine is None:
        return None

    trig = (f"業種内相対5d {rel_pct:+.1f}%" if not math.isnan(rel_pct) else "業種系列なし") + \
           (f" / rel-z={rel_z:+.2f} / rel-pct={rel_pctl:.0f}%" if not math.isnan(rel_z) else "") + \
           f" | 補助: 25d乖離{dev:+.1f}% RSI{rsi:.0f} 出来高z{feat['vol_z']:+.1f}" + \
           f" | 押し目{feat['dip_days']}日" + f" [{basis}]"

    c = Candidate(ticker=tkr, code=code, name=name, sector=sector, market=market,
                  direction=direction, engine=engine, basis=basis, nonfund=nonfund,
                  trigger_text=trig, feat=feat, hold_days=cfg.hold_days,
                  expiry_days=cfg.expiry_days, trail_days=cfg.trail_days)
    c.ftag = flow_tag(direction, sector, flow)
    c.sec_stage = flow.get("sector_stage", {}).get(sector, "-")

    # ---- 確信度スタッキング(v5.0配線) ----
    conf_pts = 3 if engine == "A" else 2  # A=高可 / S・B疑い=中上限
    c.conf_trail.append(f"エンジン上限 {'高' if engine=='A' else '中'}({engine})")
    c.conf_trail.append("単一ソースデータ(yfinance)使用 — 許容(減点なし)")
    if basis.startswith("縮退"):
        conf_pts -= 1
        c.conf_trail.append("正規化不成立→縮退 -1")

    regime = flow.get("regime", "不明")
    if regime == "全面高" and direction == "ロング" and engine == "A":
        conf_pts -= 1; c.conf_trail.append("レジーム全面高×反転ロング -1")
    if regime == "全面高" and direction == "ショート":
        conf_pts -= 1; c.conf_trail.append("レジーム全面高×戻り売り逆風 -1")
    if regime in ("全面安", "無風") and direction == "ロング" and engine == "A":
        c.conf_trail.append(f"レジーム{regime}=追い風(加点なし・タグのみ)")
    if c.ftag == "順流":
        c.conf_trail.append("順流タグ(追い風・加点なし)")

    if direction == "ロング" and feat["down_trend"]:
        conf_pts -= 1
        c.conf_trail.append("200日線下向き&割れ -1")
        c.flags.append("長期下落トレンド内 — ゲート0とクオリティ・キル要精査(バリュートラップ域)")

    stop_mult = cfg.stop_atr_mult
    if macro["vi"] is not None and macro["vi"] > cfg.vi_half_lot and feat["atr_pct"] > cfg.hivol_atr_pct:
        if not macro["sox_rebound"]:
            reject("Feasibility", f"高ボラ×VI>{cfg.vi_half_lot:.0f}×SOX反発無し", close)
            return None
        conf_pts -= 1
        c.conf_trail.append("高ボラタグ -1")
        c.flags.append("高ボラ: 広め損切り・小ロット")
        stop_mult = cfg.stop_atr_mult_hivol

    c.conf = "高" if conf_pts >= 3 else ("中" if conf_pts == 2 else "低")

    if direction == "ショート":
        c.checks += ["貸株在庫/空売り可否(バイナリ特則: iSPEED画面でOK)",
                     "信用倍率・売り残(踏み上げリスク)"]
        c.flags.append("ショート: 損失は理論上無限大 → 一段保守的ロット。逆日歩はコスト事前確定しない")
        if feat["reg_10pct_hit"]:
            c.flags.append("空売り価格規制の発動可能性(-10%接触) — 直近公表価格以下の新規売り不可/51単元以上適用")

    # ---- R床・到達確率・ネットR・出口設計 ----
    atr = feat["atr14"]
    if direction == "ロング":
        stop_raw = min(feat["low5"], close - stop_mult * atr) * (1 - cfg.stop_buffer_pct / 100)
        entry, risk = close, close - min(feat["low5"], close - stop_mult * atr) * (1 - cfg.stop_buffer_pct / 100)
        risk = entry - stop_raw
        room = feat["sma25"] - entry
    else:
        stop_raw = max(feat["high5"], close + stop_mult * atr) * (1 + cfg.stop_buffer_pct / 100)
        entry, risk = close, stop_raw - close
        room = entry - feat["sma25"]

    if risk <= 0:
        reject("R床", "リスク幅算出不能", close); return None
    if risk / entry * 100 > cfg.max_risk_width_pct:
        reject("R床", f"リスク幅{risk/entry*100:.1f}%>上限{cfg.max_risk_width_pct:.0f}%", close); return None
    if room < 2 * risk:
        reject("R床", f"値幅余地不足(余地{room/entry*100:+.1f}%<2R={2*risk/entry*100:.1f}%)", close); return None
    reach = (2 * risk) / max(atr, 1e-9)
    if reach > cfg.hold_days:
        reject("到達確率", f"2R到達≈{reach:.0f}ATR日>保有{cfg.hold_days}日", close); return None
    cost_r = (cfg.est_cost_pct / 100.0 * entry) / risk
    net2r = 2.0 - cost_r
    if net2r < cfg.net_r_floor:
        reject("ネットR床", f"概算ネットR{net2r:.2f}<{cfg.net_r_floor}(コスト過大)", close); return None

    sign = 1 if direction == "ロング" else -1
    c.entry, c.stop = _round_tick(entry), _round_tick(stop_raw)
    c.tp1, c.ref2r = _round_tick(entry + sign * risk), _round_tick(entry + sign * 2 * risk)
    c.risk_w, c.reach_days, c.net2r = risk, reach, net2r

    # ---- サイジング(リスク%拘束・ロット従属) ----
    if c.conf == "低":
        reject("確信度", "確信度低(累積減点) → 参考層へ", close)
        c.rank_score = -1
    else:
        base = cfg.risk_pct_high if c.conf == "高" else cfg.risk_pct_mid
        c.risk_pct = base * macro["lot_factor"]
        if cfg.account_equity > 0:
            c.shares = int(cfg.account_equity * c.risk_pct / 100 / risk // 100 * 100)
            if c.shares * entry < cfg.min_exec_jpy:
                reject("サイジング", f"サイズ過小(≈{c.shares*entry/1e4:.0f}万円<最小{cfg.min_exec_jpy/1e4:.0f}万円)につき見送り", close)
                c.rank_score = -1
                c.watch_reason = "サイズ過小"

    # ---- 確認リスト(iSPEED照合+クオリティ・キル) ----
    c.checks = [
        f"業種内相対乖離/RSI({rsi:.0f})/乖離率({dev:+.1f}%)の照合",
        f"クオリティ・キル5項目: {QUALITY_KILL_ITEMS} — 1つでも該当なら見送り",
        "直近下落のCF直結ニュース有無(TDnet) — 非ファンダ性の確定",
    ] + c.checks

    # ---- ランク(ローテーション時は順流押し目を優先) ----
    if c.rank_score >= 0:
        zz = 0.0 if math.isnan(rel_z) else min(abs(rel_z), 4.0)
        c.rank_score = (zz + min(max(feat["vol_z"], 0.0), 3.0) * 0.2
                        + (0.5 if basis == "正規化(業種内)" else 0.0)
                        + (0.4 if nonfund.startswith("痕跡なし") else 0.0)
                        + (0.3 if (regime == "ローテーション" and c.ftag == "順流") else 0.0))
    return c


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               cfg: Config, macro: dict, flow: dict, month: int) -> dict:
    rejects: List[dict] = []
    fired: List[Candidate] = []
    pool: List[Candidate] = []   # 参考層プール(低conf・S・B疑い・超過分)
    considered = 0
    eng_count = {"A": 0, "S": 0, "B疑い": 0}

    for _, row in universe.iterrows():
        df = ohlcv.get(str(row["ticker"]).strip())
        if df is None:
            continue
        n_rej = len(rejects)
        c = evaluate_stock(row.to_dict(), df, cfg, macro, flow, month, rejects)
        if c is None:
            if len(rejects) > n_rej:
                considered += 1
            continue
        considered += 1
        eng_count[c.engine] = eng_count.get(c.engine, 0) + 1
        if c.rank_score < 0:
            c.watch_reason = c.watch_reason or "確信度低(累積減点)"
            pool.append(c)
        else:
            fired.append(c)

    fired.sort(key=lambda x: (-CONF_ORDER[x.conf], -x.rank_score))

    # STEP4: 総オープンリスク上限
    cap = macro["risk_cap"]
    picked: List[Candidate] = []
    total_risk = 0.0
    for c in fired:
        if len(picked) < cfg.max_candidates and total_risk + c.risk_pct <= cap + 1e-9:
            picked.append(c)
            total_risk += c.risk_pct
        else:
            reason = "総リスク上限超過" if len(picked) < cfg.max_candidates else "候補数上限"
            rejects.append({"code": c.code, "name": c.name, "stage": "STEP4",
                            "reason": f"{reason} → 参考層", "close": round(c.feat["close"], 1)})
            c.watch_reason = reason
            pool.append(c)

    # 第2層: 参考監視層 — 本命ゼロならゼロ。本命と資金フロー関連のみ最大3(イナゴ列挙禁止)
    watch: List[Candidate] = []
    if picked:
        hon_secs = {c.sector for c in picked}
        flow_secs = flow.get("inflow_set", set()) | flow.get("outflow_set", set())
        related = [c for c in pool if c.sector in hon_secs or c.sector in flow_secs]
        related.sort(key=lambda x: -abs(x.rank_score))
        watch = related[: cfg.max_watch]

    stats = {
        "considered": considered,
        "rejected": len(rejects),
        "picked": len(picked),
        "watch": len(watch),
        "total_risk": total_risk,
        "risk_cap": cap,
        "by_engine": eng_count,
        "by_stage": pd.Series([r["stage"] for r in rejects]).value_counts().to_dict() if rejects else {},
    }
    return {"picked": picked, "watch": watch, "rejects": rejects, "stats": stats}
