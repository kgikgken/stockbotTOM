"""yfinance 取得層 — 旧 v7 の「粘りループ」を改修して流用。

ルール（SPEC §4）:
  - auto_adjust=False, actions=True。Yahoo の価格は分割調整済み・配当未調整。
    分割イベント列（Stock Splits）と配当列（Dividends）を保持し、整合性検査は adjust.py で行う。
  - ザラ場中は当日の未確定足を落とす。引け（既定 15:30 JST）以降なら当日足を含める。
  - 四本値のいずれかが NaN の行は落とす（NaN は比較が素通りするため後段に持ち込まない）。
  - 単一銘柄取得でも MultiIndex 列が返ることがあるので平坦化する。
  - 100→50→10→1 件とチャンクを縮小しつつ失敗分だけ再取得する。締切ガード付き。

yfinance は関数内で遅延 import する（テストと DRYRUN はネットワーク不要）。
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
PRICE_COLS = ["Open", "High", "Low", "Close"]

Downloader = Callable[[List[str], str], Optional[pd.DataFrame]]


# ------------------------------------------------------------------ cleaning
def _market_close(now_jst: pd.Timestamp, close_hhmm: str) -> pd.Timestamp:
    hh, mm = (int(x) for x in close_hhmm.split(":"))
    return now_jst.normalize() + pd.Timedelta(hours=hh, minutes=mm)


def clean_frame(df: pd.DataFrame, now_jst: pd.Timestamp, close_hhmm: str = "15:30") -> pd.DataFrame:
    """1銘柄分の日足を清掃する。

    - 欠損列は NaN/0 で補い、列順を COLS に揃える
    - 四本値に NaN がある行を落とす
    - tz を外す
    - 当日足は引け前なら落とす
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLS)
    df = df.copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in ("Dividends", "Stock Splits") else float("nan")
    df = df[COLS]
    df = df.dropna(subset=PRICE_COLS)
    if len(df) == 0:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    today = now_jst.normalize().tz_localize(None) if now_jst.tzinfo else now_jst.normalize()
    include_today = now_jst >= _market_close(now_jst, close_hhmm)
    if include_today:
        df = df[df.index <= today]
    else:
        df = df[df.index < today]
    df["Dividends"] = df["Dividends"].fillna(0.0)
    df["Stock Splits"] = df["Stock Splits"].fillna(0.0)
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def flatten_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """単一銘柄の結果が MultiIndex 列 (ticker, field) で返った場合に平坦化する。"""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    lvl0 = set(df.columns.get_level_values(0))
    lvlN = set(df.columns.get_level_values(-1))
    if ticker in lvl0:
        return df[ticker]
    if ticker in lvlN:
        return df.xs(ticker, axis=1, level=-1)
    return df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df


def extract_chunk(raw: Optional[pd.DataFrame], chunk: List[str], out: Dict[str, pd.DataFrame],
                  short: Set[str], now_jst: pd.Timestamp, close_hhmm: str, min_bars: int) -> None:
    """yf.download の生結果を銘柄別に分解して out に入れる。

    min_bars 未満しか無い銘柄は short（再取得しても回収できない）に入れる。
    """
    if raw is None or len(raw) == 0:
        return
    if len(chunk) == 1:
        t0 = chunk[0]
        try:
            df = clean_frame(flatten_single(raw, t0), now_jst, close_hhmm)
        except Exception:
            return
        if len(df) >= min_bars:
            out[t0] = df
            short.discard(t0)
        elif len(df):
            short.add(t0)
        return
    for t in chunk:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in set(raw.columns.get_level_values(0)):
                    continue
                sub = raw[t]
                if isinstance(sub.columns, pd.MultiIndex):
                    sub = sub.droplevel(0, axis=1)
            else:
                sub = raw
            df = clean_frame(sub, now_jst, close_hhmm)
        except Exception:
            continue
        if len(df) >= min_bars:
            out[t] = df
            short.discard(t)
        elif len(df):
            short.add(t)


def fetch_index(history_days: int, now_jst: Optional[pd.Timestamp] = None,
                close_hhmm: str = "15:30", primary: str = "^TPX", fallback: str = "^N225",
                attempts: int = 3, backoff: float = 3.0,
                downloader: Optional[Downloader] = None,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = print) -> Tuple[pd.DataFrame, str]:
    """指数（TOPIX、失敗時は日経225）の日足を取得する（DESIGN.md §12 / TASKS T-102）。

    戻り値: (DataFrame[COLS], 実際に取れた ticker)。両方失敗した場合は空 DataFrame と空文字列。
    """
    now_jst = now_jst or pd.Timestamp.now(tz="Asia/Tokyo")
    dl = downloader or _yf_downloader
    period = f"{max(int(history_days), 60)}d"
    for ticker in (primary, fallback):
        raw = None
        for a in range(attempts):
            try:
                raw = dl([ticker], period)
                if raw is not None and len(raw):
                    break
            except Exception:
                raw = None
            if a < attempts - 1:
                sleep(backoff * (a + 1))
        if raw is None or len(raw) == 0:
            log(f"[index] {ticker} 取得失敗")
            continue
        df = clean_frame(flatten_single(raw, ticker), now_jst, close_hhmm)
        if len(df):
            log(f"[index] source={ticker} 本数={len(df)}")
            return df, ticker
        log(f"[index] {ticker} 清掃後0本")
    log("[index] TOPIX/日経225 とも取得失敗")
    return pd.DataFrame(columns=COLS), ""


# ------------------------------------------------------------------ download
def _yf_downloader(chunk: List[str], period: str) -> Optional[pd.DataFrame]:
    import yfinance as yf  # 遅延 import

    return yf.download(
        tickers=" ".join(chunk), period=period, interval="1d",
        group_by="ticker", auto_adjust=False, actions=True,
        threads=True, progress=False,
    )


def fetch_ohlcv(tickers: Iterable[str], history_days: int, deadline_sec: float = 5400,
                now_jst: Optional[pd.Timestamp] = None, close_hhmm: str = "15:30",
                min_bars: int = 60, downloader: Optional[Downloader] = None,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = print) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """全銘柄の日足を取り切る。

    戻り値: (ticker -> DataFrame[COLS], meta)
    meta: data_total, data_ok, short(履歴不足), failed(取得失敗), elapsed_sec, rounds
    """
    tickers = list(dict.fromkeys(tickers))
    now_jst = now_jst or pd.Timestamp.now(tz="Asia/Tokyo")
    dl = downloader or _yf_downloader
    period = f"{max(int(history_days), 60)}d"
    out: Dict[str, pd.DataFrame] = {}
    short: Set[str] = set()
    t_start = time.time()
    rounds: List[dict] = []

    def time_left() -> bool:
        return (time.time() - t_start) < deadline_sec

    def download_chunk(chunk: List[str], attempts: int, backoff: float) -> Optional[pd.DataFrame]:
        for a in range(attempts):
            try:
                return dl(chunk, period)
            except Exception:
                if a == attempts - 1:
                    return None
                sleep(backoff * (a + 1))
        return None

    def run_round(label: str, remaining: List[str], chunk_size: int, attempts: int,
                  backoff: float, pause: float) -> int:
        before = len(out)
        for i in range(0, len(remaining), chunk_size):
            if not time_left():
                log(f"[fetch] 締切到達: {label} を中断")
                break
            chunk = remaining[i:i + chunk_size]
            raw = download_chunk(chunk, attempts, backoff)
            extract_chunk(raw, chunk, out, short, now_jst, close_hhmm, min_bars)
            sleep(pause)
        gained = len(out) - before
        rounds.append({"label": label, "remaining": len(remaining), "gained": gained})
        log(f"[fetch] {label}: 残{len(remaining)} → 回収{gained} (累計 {len(out)}/{len(tickers)})")
        return gained

    def pending() -> List[str]:
        return [t for t in tickers if t not in out and t not in short]

    run_round("1巡目(100件)", pending(), 100, attempts=3, backoff=3.0, pause=1.0)
    for label, size, attempts, backoff, pause in (
        ("2巡目(50件)", 50, 3, 3.0, 1.0),
        ("3巡目(10件)", 10, 3, 2.0, 0.5),
    ):
        rem = pending()
        if not rem or not time_left():
            break
        run_round(label, rem, size, attempts, backoff, pause)
    for r in range(6):
        rem = pending()
        if not rem or not time_left():
            break
        gained = run_round(f"単独{r + 1}巡目", rem, 1, attempts=2, backoff=2.0, pause=0.3)
        if gained == 0:
            break

    failed = pending()
    meta = {
        "data_total": len(tickers), "data_ok": len(out),
        "short": sorted(short), "failed": failed,
        "elapsed_sec": round(time.time() - t_start, 1), "rounds": rounds,
        "period": period, "asof": str(now_jst),
    }
    return out, meta
