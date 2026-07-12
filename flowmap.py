"""STEP1.5: 資金循環マップ(毎回必須・文脈フィルタ) — 東証33業種.

原則(v5.0): 資金循環はエンジンではなくタグ。「流入しているから買う」は禁止。
歪み(反転・PEAD)が主エンジンで、本マップは解消確率と保有期間の追い風/向かい風の評価、
および順流/逆流/中立タグの付与にのみ使う。
単一ソース(yfinance)算出のため段階判定・ブレッドスは「参考(縮退)」扱い。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def build_sector_frames(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame]):
    """終値・売買代金の (日付×銘柄) 行列と 銘柄→業種 対応を構築。"""
    closes, turns = {}, {}
    sec_of = {}
    for _, r in universe.iterrows():
        t = r["ticker"]
        df = ohlcv.get(t)
        if df is None or len(df) < 30:
            continue
        closes[t] = df["Close"]
        turns[t] = (df["Close"] * df["Volume"])
        sec_of[t] = r["sector"] or "不明"
    if not closes:
        return None, None, {}
    close_mat = pd.DataFrame(closes).sort_index()
    turn_mat = pd.DataFrame(turns).sort_index()
    return close_mat, turn_mat, sec_of


def sector_return_series(close_mat: pd.DataFrame, sec_of: dict,
                         min_members: int) -> Dict[str, pd.Series]:
    """業種ごとの等ウェイト日次リターン系列(業種指数の代理)。"""
    ret = close_mat.pct_change()
    out: Dict[str, pd.Series] = {}
    sectors = pd.Series(sec_of)
    for sec in sectors.unique():
        cols = sectors.index[sectors == sec]
        cols = [c for c in cols if c in ret.columns]
        if len(cols) < min_members:
            continue
        out[sec] = ret[cols].mean(axis=1)
    return out


def build_flow_map(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], cfg) -> dict:
    close_mat, turn_mat, sec_of = build_sector_frames(universe, ohlcv)
    empty = {"ok": False, "inflow": [], "outflow": [], "regime": "不明",
             "sector_ret": {}, "sector_stage": {}, "note": "データ不足で資金循環マップ作成不可"}
    if close_mat is None or close_mat.shape[0] < cfg.flow_window_long + 21:
        return empty

    sec_ret = sector_return_series(close_mat, sec_of, cfg.min_sector_members)
    if not sec_ret:
        return empty

    w, wl = cfg.flow_window, cfg.flow_window_long
    sectors = pd.Series(sec_of)
    rows = []
    for sec, r in sec_ret.items():
        cum5 = float((1 + r.iloc[-w:]).prod() - 1) * 100
        cum10 = float((1 + r.iloc[-wl:]).prod() - 1) * 100
        cols = [c for c in sectors.index[sectors == sec] if c in close_mat.columns]
        # 売買代金トレンド: 直近5日平均 / その前20日平均
        t = turn_mat[cols].sum(axis=1)
        t5 = float(t.iloc[-w:].mean())
        t20 = float(t.iloc[-(w + 20):-w].mean())
        turn_ratio = t5 / t20 if t20 > 0 else np.nan
        # ブレッドス(値上がり銘柄比率・5日) — 単一ソース参考値
        c5 = close_mat[cols]
        breadth = float((c5.iloc[-1] / c5.iloc[-1 - w] - 1 > 0).mean()) * 100
        rows.append({"sector": sec, "ret5": cum5, "ret10": cum10,
                     "turn_ratio": turn_ratio, "breadth": breadth, "n": len(cols)})
    fm = pd.DataFrame(rows)

    # 流入/流出: 騰落率の持続性(5日・10日同方向)＋売買代金増を加味したスコア
    fm["in_score"] = fm["ret5"] + 0.5 * fm["ret10"] + 2.0 * (fm["turn_ratio"].fillna(1) - 1).clip(-1, 1) * 3
    fm = fm.sort_values("in_score", ascending=False).reset_index(drop=True)
    inflow = fm.head(3).to_dict("records")
    outflow = fm.tail(3).iloc[::-1].to_dict("records")

    # 段階判定(参考・単一ソース): 流入側と流出側で別ルーブリック
    def stage_in(r):
        if r["ret5"] > 0 and r["turn_ratio"] and r["turn_ratio"] > 1.15 and r["breadth"] < 65:
            return "初動"
        if r["ret5"] > 0 and r["ret10"] > 0 and r["breadth"] >= 65:
            return "成熟"
        if r["ret10"] > 0 and (r["ret5"] <= 0 or (r["turn_ratio"] and r["turn_ratio"] < 0.9)):
            return "末期"
        return "不明"

    def stage_out(r):
        if r["ret5"] < 0 and r["ret10"] < 0:
            return "成熟以降"
        if r["ret5"] < 0:
            return "初動"
        return "不明"

    stages = {}
    for r in inflow:
        stages[r["sector"]] = stage_in(r)
    for r in outflow:
        stages[r["sector"]] = stage_out(r)

    # レジームタグ
    up_share = float((fm["ret5"] > 0).mean())
    spread = float(fm["ret5"].head(3).mean() - fm["ret5"].tail(3).mean())
    if up_share >= cfg.regime_up_share:
        regime = "全面高"
    elif up_share <= cfg.regime_dn_share:
        regime = "全面安"
    elif spread >= cfg.regime_rotation_spread:
        regime = "ローテーション"
    else:
        regime = "無風"

    return {
        "ok": True,
        "inflow": inflow, "outflow": outflow,
        "inflow_set": {r["sector"] for r in inflow},
        "outflow_set": {r["sector"] for r in outflow},
        "sector_stage": stages,
        "regime": regime, "up_share": up_share * 100, "spread": spread,
        "sector_ret": sec_ret,
        "note": "業種指数=構成銘柄等ウェイト代理・単一ソース参考。部門別売買動向(週次)はラグのため不使用。",
    }


def flow_tag(direction: str, sector: str, flow: dict) -> str:
    """順流/逆流/中立 — 建玉方向とセクター資金の向きの整合。"""
    if not flow.get("ok"):
        return "中立"
    if direction == "ロング":
        if sector in flow["inflow_set"]:
            return "順流"
        if sector in flow["outflow_set"]:
            return "逆流"
    else:
        if sector in flow["outflow_set"]:
            return "順流"
        if sector in flow["inflow_set"]:
            return "逆流"
    return "中立"
