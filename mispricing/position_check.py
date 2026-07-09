"""保有ポジション評価 — 「今スクリーニングしたら何点か」+ 日次ストップ/利確参照水準の再計算.

positions.csv の各銘柄を、新規候補と全く同じエンジン判定(evaluate_stock)に
本日のデータで通す。エンジンが点灯すれば、その確信度・IN/STOP/TP1/2Rが
「今日新規に評価した場合の水準」としてそのまま使える。
非点灯でも、当日のATR/直近高安値ベースの構造的ストップは必ず出す
(利確・損切り位置は日々動くため、トレーリング判断の参考値として毎回計算)。

含み損益はR倍数ではなく価格と%のみ(当初stop_priceの記録を必須にしないための割り切り)。
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import Config
from .data import fetch_ohlcv
from .indicators import compute_features
from .screen import evaluate_stock, _round_tick


def _structural_stop(feat: dict, direction: str, cfg: Config, macro: dict) -> float:
    """エンジン非点灯時でも出す、当日ATR/直近5日高安値ベースの構造的ストップ参考値."""
    atr = feat["atr14"]
    stop_mult = cfg.stop_atr_mult
    if (macro.get("vi") is not None and macro["vi"] > cfg.vi_half_lot
            and feat["atr_pct"] > cfg.hivol_atr_pct):
        stop_mult = cfg.stop_atr_mult_hivol
    close = feat["close"]
    if direction == "ロング":
        raw = min(feat["low5"], close - stop_mult * atr) * (1 - cfg.stop_buffer_pct / 100)
    else:
        raw = max(feat["high5"], close + stop_mult * atr) * (1 + cfg.stop_buffer_pct / 100)
    return _round_tick(raw)


def check_positions(pos_df: pd.DataFrame, universe: pd.DataFrame, cfg: Config,
                    macro: dict, flow: dict, month: int) -> List[dict]:
    if pos_df is None or len(pos_df) == 0:
        return []

    tickers = [str(t).strip() for t in pos_df["ticker"].tolist() if str(t).strip()]
    ohlcv, _ = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)

    uni_idx = universe.drop_duplicates("ticker").set_index("ticker") if len(universe) else None

    out: List[dict] = []
    for _, p in pos_df.iterrows():
        ticker = str(p["ticker"]).strip()
        try:
            shares = float(p.get("shares", 0))
        except Exception:
            shares = 0.0
        direction = "ショート" if shares < 0 else "ロング"

        urow = uni_idx.loc[ticker] if (uni_idx is not None and ticker in uni_idx.index) else None
        name = str(urow["name"]) if urow is not None else ticker
        sector = str(urow["sector"]) if urow is not None else ""
        market = str(urow["market"]) if urow is not None else "Prime"

        item = {"ticker": ticker, "code": ticker.replace(".T", ""), "name": name,
                "sector": sector, "direction": direction, "shares": abs(shares)}

        df = ohlcv.get(ticker)
        if df is None:
            item["error"] = "データ取得不可(コード不一致 or 上場廃止の可能性、要確認)"
            out.append(item)
            continue
        feat = compute_features(df)
        if feat is None:
            item["error"] = "データ不足(算出不可)"
            out.append(item)
            continue

        close = feat["close"]
        item["close"] = close
        try:
            entry_price = float(p["entry_price"])
            item["entry_price"] = entry_price
            sign = 1 if direction == "ロング" else -1
            item["pnl_pct"] = sign * (close / entry_price - 1) * 100
        except Exception:
            pass

        item["today_stop"] = _structural_stop(feat, direction, cfg, macro)
        item["today_sma25"] = round(feat["sma25"], 1)
        item["rel_dev_pct"] = round(feat["dev25"] * 100, 1)
        item["rsi14"] = round(feat["rsi14"], 1)

        # 流動性フィルタ自体を下回っていないか(保有中の警告として重要)
        liq_warn = None
        if close < cfg.min_price:
            liq_warn = f"株価が最低基準({cfg.min_price:.0f}円)を下回っています"
        elif feat["adv20_jpy"] < cfg.min_adv_jpy:
            liq_warn = f"ADVがスクリーニング基準(5億円)を下回っています(≈{feat['adv20_jpy']/1e8:.1f}億円)"
        item["liquidity_warn"] = liq_warn

        rejects_tmp: list = []
        cand = evaluate_stock({"ticker": ticker, "name": name, "sector": sector, "market": market},
                              df, cfg, macro, flow, month, rejects_tmp)
        item["candidate"] = cand
        if cand is not None:
            item["status_note"] = f"本日も{cand.engine}型トリガー継続 → 確信度{cand.conf}"
        elif liq_warn:
            item["status_note"] = f"本日はトリガー非点灯・{liq_warn}"
        elif rejects_tmp:
            item["status_note"] = f"本日は非点灯({rejects_tmp[-1]['reason']})"
        else:
            item["status_note"] = "本日は非点灯(反転済み、または条件外)"
        out.append(item)
    return out
