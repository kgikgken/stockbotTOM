"""stockbotTOM v6.0 — エントリーポイント。

実行順序:
  1. ユニバース読込 → 2. データ取得 → 3. 指数取得 → 4. 保有銘柄の出口判定
  → 5. パイプライン(Layer0〜5) → 6. 監視ユニバース保存 → 7. 通知
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from v6.config import load_config_v6
from v6.pipeline import run_pipeline
from v6.exits import evaluate_exit
from v6.notify import build_notification

from v6.data import fetch_ohlcv
from v6.universe import load_universe
from v6.line_send import send_line

JST = timezone(timedelta(hours=9))
POSITIONS_PATH = "positions_v6.csv"
POS_COLS = ["ticker", "shares", "entry_price", "entry_date", "stop_price",
            "setup", "partial_done", "be_moved"]


def load_positions(path: str = POSITIONS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=POS_COLS)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in POS_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def fetch_index(cfg) -> pd.DataFrame | None:
    """TOPIX(^N225で代用可)の日足。相対強度の基準に使う。"""
    if cfg.dryrun:
        n = 420
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        import numpy as np
        return pd.DataFrame({"Close": np.linspace(1000, 1100, n)}, index=idx)
    try:
        import yfinance as yf
        h = yf.Ticker("^N225").history(period="2y")
        if len(h) < 260:
            return None
        h.index = h.index.tz_localize(None) if getattr(h.index, "tz", None) else h.index
        return h[["Close"]]
    except Exception:
        return None


def main() -> dict:
    cfg = load_config_v6()
    now = datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    uni = load_universe("universe_jpx.csv")
    if cfg.dryrun:
        uni = uni.head(60)
    print(f"[1/6] universe {len(uni)}銘柄 / dryrun={cfg.dryrun}")

    ohlcv, meta = fetch_ohlcv(uni["ticker"].tolist(), cfg.history_days,
                              dryrun=cfg.dryrun, deadline_sec=5400)
    print(f"[2/6] OHLCV {meta.get('data_ok','?')}/{meta.get('data_total','?')}")

    index_daily = fetch_index(cfg)
    if index_daily is None:
        print("[3/6] ⚠指数(N225)取得失敗 — 相対強度が計算できないため中止")
        send_line("⚠stockbotTOM v6: 指数データが取得できず、相対強度が計算不能。"
                  "本日の通知は中止します。")
        return {"ok": False, "reason": "index_unavailable"}
    print(f"[3/6] 指数 {len(index_daily)}行")

    # ---- 保有銘柄の出口判定(件数上限なし) ----
    pos_df = load_positions()
    name_map = {}
    try:
        name_map = {str(r["ticker"]).strip(): str(r.get("name", ""))
                    for _, r in uni.iterrows()}
    except Exception:
        pass
    exits = []
    for _, p in pos_df.iterrows():
        tkr = str(p.get("ticker", "")).strip()
        if not tkr:
            continue
        df = ohlcv.get(tkr)
        if df is None:
            exits.append({"code": tkr.replace(".T", ""),
                          "name": name_map.get(tkr, tkr),
                          "top": (0, "データ取得不可", "要手動確認"), "metrics": {}})
            continue
        try:
            position = {
                "entry": float(p["entry_price"]), "stop": float(p["stop_price"]),
                "shares": float(p["shares"]), "entry_date": p["entry_date"],
                "partial_done": str(p.get("partial_done", "")).strip() in {"1", "TRUE", "True"},
                "be_moved": str(p.get("be_moved", "")).strip() in {"1", "TRUE", "True"},
            }
        except Exception:
            continue
        r = evaluate_exit(df, position, cfg, today)
        r["code"] = tkr.replace(".T", "")
        r["name"] = name_map.get(tkr, tkr)
        exits.append(r)
    print(f"[4/6] 保有{len(pos_df)}件 → 出口シグナル{sum(1 for e in exits if e.get('top'))}件")

    # ---- ポートフォリオ状態 ----
    positions_for_heat = []
    for _, p in pos_df.iterrows():
        tkr = str(p.get("ticker", "")).strip()
        df = ohlcv.get(tkr)
        if df is None:
            continue
        try:
            positions_for_heat.append({
                "close": float(df["Close"].dropna().iloc[-1]),
                "stop": float(p["stop_price"]), "shares": float(p["shares"])})
        except Exception:
            continue

    equity = cfg.equity
    peak = float(os.getenv("V6_PEAK_EQUITY", equity))

    res = run_pipeline(uni, ohlcv, index_daily, cfg, today,
                       positions=positions_for_heat, equity=equity, peak_equity=peak)
    c = res["counts"]
    print(f"[5/6] {c['universe']}→L0:{c['layer0']}→L1:{c['layer1']}→L2:{c['layer2']}"
          f"→L3:{c['layer3']}→L4:{c['layer4']}→通知:{c['layer5']}")

    # ---- 監視ユニバース(Layer2通過)を保存 ----
    if res["watchlist"]:
        pd.DataFrame(res["watchlist"]).to_csv(
            outdir / "watchlist.csv", index=False, encoding="utf-8-sig")
    print(f"[6/6] 監視ユニバース {len(res['watchlist'])}銘柄 → watchlist.csv")

    text = build_notification(res, exits, today, cfg)
    (outdir / f"v6_report_{today}.txt").write_text(text, encoding="utf-8")

    result = send_line(text)
    print(f"      LINE: {result}")
    return {"ok": True, "picked": c["layer5"], "line": result}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        try:
            send_line(f"⚠stockbotTOM v6 エラーで停止: {type(e).__name__}: {e}")
        except Exception:
            pass
        raise
