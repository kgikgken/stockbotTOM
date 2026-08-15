"""stockbotTOM v7.1 — エントリーポイント(次元抽出型スクリーニング)。

実行順序:
  1. ユニバース読込 → 2. データ取得 → 3. 指数取得
  → 4. パイプライン(L0〜L5) → 5. 出力保存 → 6. 通知

本ロジックはスクリーニング層のみ。エントリー価格・サイズ・出口は含まない(§0.1)。
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from v7.config import load_config_v7
from v7.pipeline import run_pipeline, DIM_NAMES
from v7.notify import build_notification
from v7.data import fetch_ohlcv
from v7.universe import load_universe
from v7.line_send import send_line

JST = timezone(timedelta(hours=9))


def fetch_index(cfg) -> pd.DataFrame | None:
    """市場レジーム判定用の指数。

    仕様書はTOPIXを指定するが、yfinanceでの安定取得を優先し、
    TOPIX(^TPX)→日経225(^N225)の順で試す。
    ※§10.4のTOPIX非定常性の問題は、②の市場ファクターを自前構築することで
      別途対処している(L0-Aの指数選択はこの問題の影響が相対的に小さい)。
    """
    if cfg.dryrun:
        import numpy as np
        # bdate_rangeはendが非営業日(土日)だと、periods指定より1件少なく返す
        # ことがある(pandas 3.0.2で確認)。nをbdate_range側の実件数に合わせる。
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=500)
        n = len(idx)
        p = np.linspace(2000, 2900, n)
        return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99,
                             "Close": p, "Volume": np.full(n, 1e9)}, index=idx)
    try:
        import yfinance as yf
        for sym in ("^TPX", "^N225"):
            try:
                h = yf.Ticker(sym).history(period="3y")
                if len(h) >= 300:
                    if getattr(h.index, "tz", None) is not None:
                        h.index = h.index.tz_localize(None)
                    print(f"      指数: {sym} ({len(h)}行)")
                    return h[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:
                continue
    except Exception:
        pass
    return None


def main() -> dict:
    cfg = load_config_v7()
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
        print("[3/6] ⚠指数取得失敗 — レジーム判定不能のため中止")
        send_line("⚠stockbotTOM v7.1: 指数が取得できずレジーム判定不能。本日の通知は中止します。")
        return {"ok": False, "reason": "index_unavailable"}
    print(f"[3/6] 指数 {len(index_daily)}行")

    res = run_pipeline(uni, ohlcv, index_daily, cfg, today)
    c = res["counts"]
    print(f"[4/6] レジーム {res['regime']['state']} / "
          f"{c['universe']}→L1:{c['l1']}→採点:{c['scored']}"
          f"→本命:{c['main']}/監視:{c['watch']}")

    # ---- 次元スコアの分布を保存(Phase 2のヒストグラム確認用・§4.2.1) ----
    # res["main"]/res["watch"]は表示用に上位max_output/max_watch_output件へ
    # 切られた部分集合。ここから保存すると本命/監視への選抜バイアスが乗り、
    # Phase 2のヒストグラムもverify_level1.pyの独立性検定も歪む(選抜条件が
    # 5次元の和と相関するコライダーになるため)。L1通過・採点済みの全件(除外含む)
    # であるall_scoredを使う。
    rows = []
    for cand in res.get("all_scored", []):
        r = {"date": today, "code": cand.code, "name": cand.name,
             "sector": cand.sector, "dcs": round(cand.dcs, 2),
             "coverage": round(cand.coverage, 3), "category": cand.category}
        for d, v in cand.dims.items():
            r[f"dim{d}"] = None if v is None else round(v, 4)
        # サブ指標(4-a等)。次元合成値だけでは「何が足を引っ張っているか」が
        # 分からないため、Phase 2の較正判断(§13)にはこの粒度が要る。
        # 銘柄によって base 未検出等で欠けるキーがあるため、DataFrame化の際は
        # 全銘柄のキー和集合で列が作られ、無い銘柄はNaNになる(想定通り)。
        for _, parts in cand.parts.items():
            for key, val in parts.items():
                r[f"part_{key}"] = None if val is None else round(val, 4)
        r["missing"] = ",".join(str(d) for d in cand.missing)
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "dimension_scores.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"[5/6] 次元スコア {len(rows)}件 → dimension_scores.csv")

    text = build_notification(res, today, cfg)
    (outdir / f"v7_report_{today}.txt").write_text(text, encoding="utf-8")

    result = send_line(text)
    print(f"[6/6] LINE: {result}")
    return {"ok": True, "main": c["main"], "line": result}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        try:
            send_line(f"⚠stockbotTOM v7.1 エラーで停止: {type(e).__name__}: {e}")
        except Exception:
            pass
        raise
