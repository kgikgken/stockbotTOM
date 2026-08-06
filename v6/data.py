"""v6データ層 — 実データ取得(粘りループ・締切ガード付き)とDRYRUN合成データ。
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
        取引時間中(ザラ場)は必ず切り捨てて前営業日までを使う(_clean内で実施)。
        切らないと『同じ日に数分後リトライしただけでギャップ率・ATR・出来高比・
        トレンド判定が全部変わる』事故になる(2026-07-27、手動実行時に発覚)。

        ★2026-07-27 追加修正: 「今日の日付なら無条件で除外」だと、引け後(15:30 JST以降)に
        実行しても今日の確定足が永遠に見えず、引け後実行が朝の実行と同じ結果になる事故が
        起きた(同日に場前・場後で回して同一銘柄が出た)。引けの時刻で判定を分ける。
          - 現在時刻が今日15:30より前 → 今日の足は未確定 → 除外(従来通り)
          - 現在時刻が今日15:30以降   → 今日の足は確定済み → 含める
        """
        if raw is None or len(raw) == 0:
            return
        now_jst = pd.Timestamp.now(tz="Asia/Tokyo")
        today_jst = now_jst.normalize().tz_localize(None)
        market_close = now_jst.normalize() + pd.Timedelta(hours=15, minutes=30)
        include_today = now_jst >= market_close   # 引け後なら今日の足も確定済みとして使う

        def _clean(df):
            df = df.dropna(how="all")
            if len(df) == 0:
                return df
            idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
            if include_today:
                return df[idx.normalize() <= today_jst]
            return df[idx.normalize() < today_jst]

        def _flatten_single(df, ticker):
            """1銘柄取得でも yfinance が group_by="ticker" 指定によりMultiIndex列
            (ticker, field) を返すことがある(バージョン依存)。列がMultiIndexなら
            平らな 'Close'/'Open'等 の列に直す。既に平らならそのまま返す。
            これを怠ると後段の df["Close"] が KeyError になる
            (2026-08-06、本番の単独リトライ巡で発覚)。
            """
            if not isinstance(df.columns, pd.MultiIndex):
                return df
            lvl0 = set(df.columns.get_level_values(0))
            lvl1 = set(df.columns.get_level_values(-1))
            if ticker in lvl0:
                return df[ticker]
            if ticker in lvl1:
                return df.xs(ticker, axis=1, level=-1)
            # ティッカー名がどちらの階層にも見つからない場合の保険。
            # 通常は起きないが、原因不明のまま欠測にするよりは1階層落として救う。
            return df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df

        if len(chunk) == 1:
            t0 = chunk[0]
            try:
                df = _clean(_flatten_single(raw, t0))
            except Exception:
                return
            if len(df) >= 60:
                out[t0] = df
            elif len(df):
                short.add(t0)
            return
        for t in chunk:
            try:
                sub = raw[t]
                if isinstance(sub.columns, pd.MultiIndex):
                    sub = sub.droplevel(0, axis=1) if sub.columns.nlevels > 1 else sub
                df = _clean(sub)
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

    # ★引け後実行なのに、Yahoo側の反映が遅れて今日の足がまだ来ていないケースを検知する。
    # これに気づかないと「引け後に実行したのに朝と同じ結果」という事故が再発する
    # (今度は_cleanのバグではなく、データ提供元側の遅延が原因で)。
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo")
    market_close = now_jst.normalize() + pd.Timedelta(hours=15, minutes=30)
    today_bar_expected = now_jst >= market_close
    today_bar_present = None
    if today_bar_expected and out:
        today_date = now_jst.normalize().tz_localize(None)
        sample = list(out.values())[: min(50, len(out))]
        have = sum(1 for df in sample if len(df) and df.index[-1].normalize() == today_date)
        today_bar_present = have / len(sample) if sample else 0.0

    meta.update({
        "pass_log": pass_log,
        "short_history": len(short),
        "failed": len([t for t in tickers if t not in out and t not in short]),
        "today_bar_expected": today_bar_expected,
        "today_bar_present_ratio": today_bar_present,
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


def _mk_v6(seed: int, n: int, mode: str) -> pd.DataFrame:
    """v6の検証に意味のあるパターンを生成する。

    v6は「週足ステージ2 → トレンドテンプレート → 日足ステージ1 → セットアップ」という
    直列フィルターなので、旧v5の材料反応(ギャップ)やTOBパターンは検証に使えない。
    代わりに、長期上昇トレンドの中で VCP / 押し目 / ピボットブレイク を作り分ける。
    """
    rng = np.random.default_rng(seed)
    idx = _bdate_index(n)

    if mode == "vcp":
        # 長期上昇 → 収縮する3つの押し → 出来高枯渇(VCPの理想形)
        seg = [np.linspace(400, 900, n - 130)]
        p = 900.0
        for depth, ln in [(0.14, 18), (0.08, 15), (0.035, 12)]:
            lo = p * (1 - depth)
            seg.append(np.linspace(p, lo, ln))
            p = lo * 1.10
            seg.append(np.linspace(lo, p, ln))
        rest = n - sum(len(x) for x in seg)
        seg.append(np.linspace(p, p * 1.01, max(1, rest)))
        price = np.concatenate(seg)[:n]
        vol = np.concatenate([np.full(n - 45, 3e6), np.full(45, 8e5)])[:n]

    elif mode == "pullback":
        # 長期上昇 → 直近で「浅い」押し(出来高減少を伴う)。
        # 押しが深すぎると5日線が40日線を下抜けて日足ステージ4になり、
        # 押し目の前提(ステージ1)を自ら壊してしまうため深さは6%程度に留める。
        # ★押し目は「日足ステージ1(5>20>40)」と「高値から3〜12%下」を同時に満たす
        # 必要があり、この2つが両立する範囲は狭い。押しが長引くと5日線が20日線を
        # 下抜けてステージ1でなくなり、逆に十分回復すると深さが3%を割る。
        # 総当たり検証の結果、複利で上昇するトレンド中の「短い押し(3〜8日)で
        # まだ回復しきっていない」局面が該当するため、その形を作る。
        nb = n - 5
        base = 500 * np.cumprod(np.full(nb, 1.004))     # 複利0.4%/日の上昇
        hi = base[-1]
        dip = np.linspace(hi, hi * 0.955, 5)            # 4.5%の押しを5日で
        price = np.concatenate([base, dip])[:n]
        vol = np.concatenate([np.full(n - 5, 3e6), np.full(5, 1.0e6)])[:n]

    elif mode == "pivot":
        # ベース形成 → 当日に出来高を伴って上抜け
        seg = [np.linspace(400, 880, n - 60), np.full(58, 880.0) * (1 + rng.normal(0, 0.008, 58))]
        seg.append(np.array([895.0, 900.0]))
        price = np.concatenate(seg)[:n]
        vol = np.concatenate([np.full(n - 2, 2e6), np.array([6e6, 7e6])])[:n]

    elif mode == "downtrend":
        price = np.linspace(900, 400, n)
        vol = np.full(n, 2e6)

    else:  # normal: 緩やかなランダムウォーク(どのレイヤーも通らない想定)
        ret = rng.normal(0.0002, 0.016, n)
        price = 700.0 * np.cumprod(1 + ret)
        vol = np.abs(rng.normal(2e6, 4e5, n))

    price = np.maximum(price, 1.0)
    # ★高値/安値を一律倍率にすると終値が常にレンジのちょうど中央になり、
    # 「終値が高値圏(上位60%)」を要求するピボット条件が構造的に成立しなくなる。
    # 上昇日は高値寄り、下落日は安値寄りで引けるよう、前日比で寄せる。
    prev = np.concatenate([[price[0]], price[:-1]])
    up = price >= prev
    rng_w = price * 0.028   # 日中値幅 約2.8%(実際の中小型株に近い水準)
    high = np.where(up, price + rng_w * 0.25, price + rng_w * 0.75)
    low = np.where(up, price - rng_w * 0.75, price - rng_w * 0.25)
    return pd.DataFrame({
        "Open": price * (1 + rng.normal(0, 0.002, n)),
        "High": high, "Low": low,
        "Close": price, "Volume": vol}, index=idx)


def _synthetic(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """DRYRUN用の合成データ。v6のレイヤー構造を検証できる配分にする。

    先頭6銘柄に各パターンを1つずつ配置し、残りはnormal(通過しない)にする。
    これによりファネルの各段が機能しているかを毎回確認できる。
    """
    modes = ["vcp", "pullback", "pivot", "vcp", "downtrend", "pullback"]
    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:60]):
        m = modes[i] if i < len(modes) else "normal"
        out[t] = _mk_v6(seed=511 + i, n=420, mode=m)
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
    ratio = meta.get("today_bar_present_ratio")
    if meta.get("today_bar_expected") and ratio is not None and ratio < 0.5:
        bits.append(f"⚠引け後実行だが本日確定足が銘柄の{ratio:.0%}にしか反映されていない"
                    "(Yahoo側の更新待ち。結果が前営業日と同一の可能性)")
    return " / ".join(bits)


def coverage_detail(meta: dict) -> str:
    """各巡でどれだけ回収できたかの内訳(ログ確認用・レポートには出さない)。"""
    lines = []
    for label, gained, total in (meta.get("pass_log") or []):
        lines.append(f"  {label}: +{gained}件 (累計{total}件)")
    return "\n".join(lines)
