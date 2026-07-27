"""topdown用データ層 — 実データはmomentum/mispricingの汎用フェッチ(2巡目リトライ込み)を再利用。
DRYRUN合成データはギャップ型・ブレイク型・押し目型・S高急騰型・TOB型・通常を作り分ける。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import time  # 実フェッチのバックオフ用(自立版)


def fetch_ohlcv(tickers: List[str], history_days: int, dryrun: bool = False,
                deadline_sec: float = 6000) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """deadline_sec: 取得に使ってよい上限秒数(既定100分)。呼び出し側の実行時間予算に合わせて渡す。"""
    if dryrun:
        return _synthetic(tickers)
    out, meta = _fetch_ohlcv_real(tickers, history_days, deadline_sec=deadline_sec)
    meta["fetch_failures"] = [t for t in tickers if t not in out]  # 取得失敗の記録(旧momentum仕様を継承)
    return out, meta


# ---- 実フェッチ(自立版: 旧mispricing/data.pyから逐語移植・2巡目リトライ込み) ----
def _fetch_ohlcv_real(tickers: List[str], history_days: int,
                      cap_singles: int = 400,
                      deadline_sec: float = 6000) -> Tuple[Dict[str, pd.DataFrame], dict]:
    import yfinance as yf

    period = f"{max(history_days, 400)}d"

    def _download_chunk(chunk: List[str], attempts: int, base_backoff: float):
        for attempt in range(attempts):
            try:
                return yf.download(
                    tickers=" ".join(chunk), period=period, interval="1d",
                    group_by="ticker", auto_adjust=False, actions=False,
                    threads=True, progress=False,
                )
            except Exception:
                if attempt == attempts - 1:
                    return None
                time.sleep(base_backoff * (attempt + 1))
        return None

    # 欠落の内訳を分けて記録する(2026-07-24)。
    #   short  : 応答はあったが履歴が60日未満(新規上場等) → 何度再取得しても回収できない
    #   failed : 応答自体が無い/壊れている        → 再取得で回収できる可能性がある
    # 従来はこの2つが同じ「欠落」に混ざり、被覆率が低い原因を判別できなかった。
    short: set = set()

    def _extract(raw, chunk: List[str], out: Dict[str, pd.DataFrame]):
        """当日の日足は市場が開いている間ずっと動き続ける未確定値のため、
        必ず切り捨てて前営業日までを使う(_clean内で実施)。切らないと
        『同じ日に数分後リトライしただけでギャップ率・ATR・出来高比・
        トレンド判定が全部変わる』事故になる(2026-07-27、手動実行時に発覚)。
        自動実行(cron 7:30 JST)は前場開始前なので通常は影響しないが、
        手動実行(workflow_dispatch)は市場が開いている時間にも走らせられるため必須。
        """
        if raw is None or len(raw) == 0:
            return
        today_jst = pd.Timestamp.now(tz="Asia/Tokyo").normalize().tz_localize(None)

        def _clean(df):
            df = df.dropna(how="all")
            if len(df) == 0:
                return df
            idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
            # 当日分(未確定の可能性がある最終行)は落とす。前営業日までを使う。
            return df[idx.normalize() < today_jst]

        if len(chunk) == 1:
            df = _clean(raw)
            if len(df) >= 60:
                out[chunk[0]] = df
            elif len(df):
                short.add(chunk[0])
            return
        for t in chunk:
            try:
                df = _clean(raw[t])
            except Exception:
                continue
            if len(df) >= 60:
                out[t] = df
                short.discard(t)
            elif len(df):
                short.add(t)

    # ★2026-07-27: 「取れた銘柄だけで判定する」から「粘って全銘柄を取り切る」へ方針転換。
    # 取得成否が実行のたびに変わり、同じ銘柄が"見えたり見えなかったり"して出力が
    # 安定しない問題があった(78%〜97%でばらつき)。時間をかけてよいとの判断のもと、
    # 失敗が尽きるまでチャンクを縮小し続けるループに変更する。
    #   1巡目: 100件チャンク
    #   2巡目: 50件チャンク(1巡目の失敗のみ)
    #   3巡目: 10件チャンク(2巡目の失敗のみ)
    #   4巡目以降: 1件ずつ、成功が止まるまで繰り返す(最大 max_rounds 回)
    # yfinance側の恒久的な欠落(上場廃止・コード変更等)は何度やっても回収できないため、
    # "直前の巡で1件も増えなかったら打ち切る"ことで無限ループを避ける。
    # ★締切(既定100分)を過ぎたら粘るのをやめる。9:00の寄り付きに配信を間に合わせるため。
    # 時間切れの場合は「そこまでに取れた分」で判定を進める(取れた分は全部使う。
    # 全か無かではなく、常に最善の被覆率で動かす)。
    t_start = time.time()

    def _time_left() -> bool:
        return (time.time() - t_start) < deadline_sec

    out: Dict[str, pd.DataFrame] = {}
    pass_log: List[tuple] = []

    def _round(remaining: List[str], chunk_size: int, attempts: int, backoff: float, pause: float):
        before = len(out)
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i:i + chunk_size]
            raw = _download_chunk(chunk, attempts=attempts, base_backoff=backoff)
            _extract(raw, chunk, out)
            time.sleep(pause)
        return len(out) - before

    remaining = list(tickers)
    gained = _round(remaining, 100, attempts=3, backoff=3, pause=1.0)
    pass_log.append(("1巡目(100件)", gained, len(out)))

    for label, chunk_size, attempts, backoff, pause in (
        ("2巡目(50件)", 50, 2, 4, 1.5),
        ("3巡目(10件)", 10, 2, 3, 1.0),
    ):
        if not _time_left():
            pass_log.append((f"{label}: 締切のため未実施", 0, len(out)))
            break
        remaining = [t for t in tickers if t not in out and t not in short]
        if not remaining:
            break
        gained = _round(remaining, chunk_size, attempts, backoff, pause)
        pass_log.append((label, gained, len(out)))

    # 4巡目以降: 1件ずつ。増分が0になったら(=もう回収できるものが無い)打ち切る。
    max_rounds = 6
    for r in range(max_rounds):
        if not _time_left():
            pass_log.append((f"単独{r + 1}巡目以降: 締切のため打ち切り", 0, len(out)))
            break
        remaining = [t for t in tickers if t not in out and t not in short]
        if not remaining:
            break
        gained = 0
        for t in remaining:
            raw = _download_chunk([t], attempts=2, base_backoff=2)
            before = len(out)
            _extract(raw, [t], out)
            gained += len(out) - before
            time.sleep(0.3)
        pass_log.append((f"単独{r + 1}巡目", gained, len(out)))
        if gained == 0:
            break   # これ以上粘っても回収できない(恒久的な欠落)

    meta = _coverage(tickers, out, source="yfinance(Yahoo Finance) 単一ソース")
    meta.update({
        "pass_log": pass_log,
        "short_history": len(short),
        "failed": len([t for t in tickers if t not in out and t not in short]),
    })
    return out, meta


def _coverage(tickers, out, source):
    total = len(tickers)
    ok = len(out)
    return {
        "data_total": total,
        "data_ok": ok,
        "data_coverage": (ok / total) if total else 0.0,
        "source": source,
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
    }


def _bdate_index(n: int) -> pd.DatetimeIndex:
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:
        end -= pd.tseries.offsets.BDay(1)
    return pd.bdate_range(end=end, periods=n)


def _mk_series_topdown(seed: int, n: int, mode: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _bdate_index(n)

    if mode == "gap_catalyst":
        # 平常運転+2営業日前に決算等想定の単日+6%ギャップ+出来高3倍(カタリストの価格的痕跡)
        ret = rng.normal(0.0002, 0.013, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(1_500_000, 3_000_000, n).astype(float)
        gp = n - 2
        close[gp] = float(close[gp - 1]) * 1.06
        close[gp + 1:] = close[gp] * (1 + rng.normal(0.001, 0.010, n - gp - 1))
        v[gp] = v[gp - 1] * 3.0
    elif mode == "breakout":
        # 上昇トレンド+当日20日高値ブレイク+出来高1.8倍
        t = np.arange(n)
        trend = 1000.0 * np.exp(0.0008 * t)
        close = trend * (1 + rng.normal(0, 0.009, n))
        v = rng.integers(1_200_000, 2_500_000, n).astype(float)
        past_max20 = float(close[-21:-1].max())
        close[-1] = past_max20 * 1.015
        v[-1] = v[-2] * 1.8
    elif mode == "pullback":
        # momentum凍結5ゲートを満たす押し目(state_a相当の形)
        t = np.arange(n)
        trend = 1000.0 * np.exp(0.0011 * t)
        close = trend * (1 + rng.normal(0, 0.004, n))
        v = rng.integers(800_000, 1_800_000, n).astype(float)
        base = float(close[-9])
        decline = np.linspace(1.0, 0.975, 7)
        bounce = np.array([0.975 * 1.006, 0.975 * 1.012])
        close[-9:] = base * np.concatenate([decline, bounce])
    elif mode == "spike":
        # 前日比+16%の急騰済み(寄り天リスク→監視格下げ対象)
        ret = rng.normal(0.0002, 0.014, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(1_500_000, 3_000_000, n).astype(float)
        close[-1] = float(close[-2]) * 1.16
        v[-1] = v[-2] * 4.0
    elif mode == "tob_pattern":
        ret = rng.normal(0.0002, 0.016, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(1_000_000, 2_000_000, n).astype(float)
        jd = -40
        close[jd] = float(close[jd - 1]) * 1.28
        close[jd + 1:] = float(close[jd])
        v[jd] = float(v[jd - 5:jd].mean()) * 5.0
    else:
        ret = rng.normal(0.0002, 0.014, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(600_000, 1_500_000, n).astype(float)

    hl = np.full(n, 0.006)
    if mode == "tob_pattern":
        hl[-39:] = 0.0008
    o = close * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 1, n)) * hl)
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 1, n)) * hl)
    if mode == "pullback":
        o[-1] = close[-1] * 0.994; l[-1] = close[-1] * 0.990; h[-1] = close[-1] * 1.002
    if mode == "gap_catalyst":
        # ★ギャップ足に現実的な日中値幅を与える(寄りで窓を開け、そこから上昇して引ける形)。
        # これが無いとゾーン幅・損切り幅の検証が意味を持たない。
        gp = n - 2
        prev_c = float(close[gp - 1])
        o[gp] = prev_c * 1.045          # 窓を開けて寄る
        l[gp] = o[gp] * 0.995           # 寄り後の押しが日中安値
        h[gp] = prev_c * 1.065          # 日中高値
        close[gp] = prev_c * 1.060      # 高値近辺で引ける
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": v}, index=idx)


def _synthetic(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    # 0,1: gap(同一セクター内2件→セクターキャップ検証) / 2: breakout / 3: pullback / 4: spike / 5: tob
    modes = ["gap_catalyst", "gap_catalyst", "breakout", "pullback", "spike", "tob_pattern"] + ["normal"] * 6
    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:60]):
        out[t] = _mk_series_topdown(seed=511 + i, n=420, mode=modes[i % len(modes)] if i < 12 else "normal")
    meta = {
        "data_total": len(tickers[:60]), "data_ok": len(out),
        "data_coverage": 1.0 if tickers else 0.0,
        "source": "SYNTHETIC (DRYRUN・実データではない)",
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
        "dryrun": True,
    }
    return out, meta


def fetch_market_indices(dryrun: bool = False) -> dict:
    """STEP1地合い用の指数群を取得。欠落は欠落として返す(データ欠落時ルール: 暫定明記)。
    戻り値: {name: {"last": float, "chg1d_pct": float}} + {"n225_close_series": Series|None}"""
    symbols = {"SPX": "^GSPC", "DOW": "^DJI", "NASDAQ": "^IXIC", "SOX": "^SOX",
               "VIX": "^VIX", "USDJPY": "JPY=X", "N225": "^N225"}
    if dryrun:
        rng = np.random.default_rng(77)
        idx = _bdate_index(30)
        n225 = pd.Series(40000 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 30))), index=idx)
        return {
            "SPX": {"last": 6100.0, "chg1d_pct": 0.6}, "DOW": {"last": 44500.0, "chg1d_pct": 0.3},
            "NASDAQ": {"last": 20500.0, "chg1d_pct": 0.9}, "SOX": {"last": 5600.0, "chg1d_pct": 1.4},
            "VIX": {"last": 17.5, "chg1d_pct": -3.0}, "USDJPY": {"last": 152.3, "chg1d_pct": 0.2},
            "N225": {"last": float(n225.iloc[-1]), "chg1d_pct": 0.4},
            "n225_close_series": n225, "missing": [], "synthetic": True,
        }

    import yfinance as yf
    out = {"missing": [], "synthetic": False, "n225_close_series": None}
    try:
        raw = yf.download(tickers=" ".join(symbols.values()), period="60d", interval="1d",
                          group_by="ticker", auto_adjust=False, actions=False, threads=True, progress=False)
    except Exception:
        raw = None
    for name, sym in symbols.items():
        try:
            c = raw[sym]["Close"].dropna()
            if len(c) < 6:
                raise ValueError("too short")
            out[name] = {"last": float(c.iloc[-1]),
                         "chg1d_pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100)}
            if name == "N225":
                out["n225_close_series"] = c
        except Exception:
            out["missing"].append(name)
    return out


def coverage_note(meta: dict) -> str:
    """被覆率の内訳を1行にする。低い被覆率の原因が回収可能な失敗か履歴不足かを見分ける用。"""
    bits = []
    if meta.get("short_history"):
        bits.append(f"履歴60日未満 {meta['short_history']}件(回収不能)")
    if meta.get("failed"):
        bits.append(f"取得失敗 {meta['failed']}件(粘っても回収できなかった=恒久欠落の可能性)")
    rec = sum(g for _, g, _ in (meta.get("pass_log") or [])[1:])   # 1巡目以降の増分合計
    if rec:
        bits.append(f"再取得で回収 {rec}件")
    return " / ".join(bits)


def coverage_detail(meta: dict) -> str:
    """各巡でどれだけ回収できたかの内訳(ログ確認用・レポートには出さない)。"""
    lines = []
    for label, gained, total in (meta.get("pass_log") or []):
        lines.append(f"  {label}: +{gained}件 (累計{total}件)")
    return "\n".join(lines)
