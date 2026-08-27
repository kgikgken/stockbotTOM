"""JPX 公開リスト（SPEC §4 ソース）。

1. 上場銘柄一覧 data_j.xls
   列: 日付, コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分, 17業種コード, 17業種区分, 規模コード, 規模区分
   取得に失敗したら旧リポジトリの universe_jpx.csv を種として使う。

2. 決算発表予定日（T-104）… 公開ページ
   https://www.jpx.co.jp/listing/event-schedules/financial-announcement/index.html
   に「◯年◯月に期末を迎えた会社」ごとの Excel リンクが並ぶ（直近1〜2ヶ月分のローリング
   更新。完全網羅ではない。詳細は docs/DATA_SOURCES.md）。fetch_earnings_schedule() が
   ページからリンクを都度抽出して全件ダウンロード・結合する（URL のハッシュ部分は毎回
   変わるためハードコードしない）。reference/earnings_schedule.csv (code,date[,name]) を
   読む関数は load_earnings_schedule()。

3. 上場廃止銘柄一覧（T-104）… 公開ページ
   https://www.jpx.co.jp/listing/stocks/delisted/index.html
   （当年）＋ページ内バックナンバー（過去11年分、archives-NN.html）。HTML 表そのもの。
   fetch_delistings() がバックナンバーの href をページから抽出して取得・結合する。
   reference/delistings.csv (code,name,date,reason) を読む関数は load_delistings()。
   生存バイアスの上限評価（SPEC §4 緩和策 1）に使う。
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

JPX_HOST = "https://www.jpx.co.jp"  # 既定 URL は config.py の JPX_*_URL_DEFAULT 側で持つ

Getter = Callable[[str, int], bytes]

_ARCHIVE_OPTION_RE = re.compile(r'<option value="([^"]+)"[^>]*>(\d{4})年</option>')
_XLSX_HREF_RE = re.compile(r'href="([^"]+\.xlsx)"')

LISTED_COLS = ["date", "code", "ticker", "name", "market", "sector33_code", "sector33",
               "sector17_code", "sector17", "size_code", "size", "is_equity"]

_JP_COLS = {
    "日付": "date", "コード": "code", "銘柄名": "name", "市場・商品区分": "market",
    "33業種コード": "sector33_code", "33業種区分": "sector33",
    "17業種コード": "sector17_code", "17業種区分": "sector17",
    "規模コード": "size_code", "規模区分": "size",
}

# 株式以外（市場・商品区分の文字列で判定）
NON_EQUITY_PATTERNS = ("ETF", "ETN", "REIT", "インフラファンド", "ベンチャーファンド",
                       "カントリーファンド", "出資証券", "PRO Market", "PRO MARKET")


def norm_ticker(code: str) -> str:
    """'1301' / '130A' / '1301T' / '1301.T' → '1301.T'。5桁コード（優先株等）はそのまま返す。"""
    s = str(code).strip().upper()
    if s.endswith(".T"):
        return s
    m = re.fullmatch(r"(\d{3}[0-9A-Z])T", s)
    if m:
        return m.group(1) + ".T"
    if re.fullmatch(r"\d{3}[0-9A-Z]", s):
        return s + ".T"
    return s


def is_equity_market(market: str) -> bool:
    m = str(market)
    if any(p in m for p in NON_EQUITY_PATTERNS):
        return False
    return ("内国株式" in m) or ("外国株式" in m) or m in ("Prime", "Standard", "Growth")


def normalize_listed(raw: pd.DataFrame) -> pd.DataFrame:
    """JPX の列名（日本語）または旧 universe_jpx.csv の列名を LISTED_COLS に揃える。"""
    df = raw.copy()
    df = df.rename(columns={k: v for k, v in _JP_COLS.items() if k in df.columns})
    if "ticker" in df.columns and "code" not in df.columns:
        df["code"] = df["ticker"].astype(str).str.replace(".T", "", regex=False)
    if "sector" in df.columns and "sector33" not in df.columns:
        df["sector33"] = df["sector"]
    for c in LISTED_COLS:
        if c not in df.columns:
            df[c] = "" if c != "is_equity" else True
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["ticker"] = df["code"].map(norm_ticker)
    df["is_equity"] = df["market"].map(is_equity_market) & (df["code"].str.len() == 4)
    if df["date"].astype(str).str.len().gt(0).any():
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
    return df[LISTED_COLS].drop_duplicates(subset=["code"]).reset_index(drop=True)


def load_listed(source: str | Path, timeout: int = 60) -> pd.DataFrame:
    """URL または ローカルの .xls/.xlsx/.csv から上場銘柄一覧を読み、正規化して返す。"""
    src = str(source)
    if src.startswith("http://") or src.startswith("https://"):
        import requests  # 遅延 import

        r = requests.get(src, timeout=timeout, headers={"User-Agent": "stockbotTOM/0.2"})
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content))
    elif src.lower().endswith((".xls", ".xlsx")):
        raw = pd.read_excel(src)
    else:
        raw = pd.read_csv(src, dtype=str).fillna("")
    return normalize_listed(raw)


def load_listed_with_fallback(url: str, seed_csv: Path, log=print) -> tuple[pd.DataFrame, str]:
    """JPX から取得し、失敗したら種 CSV を使う。戻り値: (listed, source_label)"""
    try:
        df = load_listed(url)
        if len(df) > 1000:
            return df, "jpx"
        log(f"[listed] JPX の行数が少なすぎる({len(df)}) → 種CSVにフォールバック")
    except Exception as e:  # ネットワーク・書式変更
        log(f"[listed] JPX 取得失敗: {type(e).__name__}: {e} → 種CSVにフォールバック")
    return load_listed(seed_csv), "seed"


def _http_get_bytes(url: str, timeout: int) -> bytes:
    import requests  # 遅延 import

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "stockbotTOM/0.2"})
    r.raise_for_status()
    return r.content


def _abs_url(href: str, host: str = JPX_HOST) -> str:
    return href if href.startswith("http://") or href.startswith("https://") else host + href


def _fmt_code(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip().upper()


def _delisted_table_from_html(html: str) -> pd.DataFrame:
    """上場廃止銘柄一覧ページの HTML 表を [code, name, date, reason] に正規化する。"""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception:
        return pd.DataFrame(columns=["code", "name", "date", "reason"])
    need = {"上場廃止日", "銘柄名", "コード", "上場廃止理由"}
    for t in tables:
        if need.issubset(set(t.columns)):
            t = t.rename(columns={"上場廃止日": "date", "銘柄名": "name",
                                  "コード": "code", "上場廃止理由": "reason"})
            t["code"] = t["code"].map(_fmt_code)
            t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            return t[["code", "name", "date", "reason"]]
    return pd.DataFrame(columns=["code", "name", "date", "reason"])


def fetch_delistings(url: str, back_years: int = 11,
                     timeout: int = 60, getter: Optional[Getter] = None,
                     log=print) -> pd.DataFrame:
    """上場廃止銘柄一覧を JPX から取得する（当年＋バックナンバー、T-104）。

    ページ内の <select class="backnumber"> から実際の href を抽出して辿る
    （archives-NN.html という命名はハードコードせず、都度ページから読む）。
    過去11年分が上限（JPX の掲載仕様）。取得失敗（ネットワーク・書式変更）は
    例外を投げず、警告ログを出して取得できた分だけ返す（全滅なら空 DataFrame）。
    戻り値: DataFrame[code, name, date, reason]
    """
    get = getter or _http_get_bytes
    cols = ["code", "name", "date", "reason"]
    try:
        html = get(url, timeout).decode("utf-8", errors="replace")
    except Exception as e:
        log(f"[delistings] 現在ページ取得失敗: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=cols)

    frames = [_delisted_table_from_html(html)]
    options = _ARCHIVE_OPTION_RE.findall(html)
    archive_hrefs = [href for href, _year in options if _abs_url(href) != url]
    for href in archive_hrefs[: max(back_years - 1, 0)]:
        full = _abs_url(href)
        try:
            html2 = get(full, timeout).decode("utf-8", errors="replace")
            frames.append(_delisted_table_from_html(html2))
        except Exception as e:
            log(f"[delistings] {full} 取得失敗: {type(e).__name__}: {e} → スキップ")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df = df.dropna(subset=["code", "date"])
    df = df[(df["code"] != "") & (df["date"] != "")]
    return df.drop_duplicates(subset=["code", "date"]).sort_values(["date", "code"]).reset_index(drop=True)


def _earnings_table_from_xlsx(content: bytes) -> pd.DataFrame:
    """決算発表予定日 xlsx（1シート、先頭は見出し行）を [date, code, name] に正規化する。

    見出し行の位置は将来のレイアウト変更に備えて固定インデックスにせず、
    2列目のセルに「コード」が現れる行を探して決める。
    """
    cols = ["date", "code", "name"]
    try:
        raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    except Exception:
        return pd.DataFrame(columns=cols)
    if raw.shape[1] < 3:
        return pd.DataFrame(columns=cols)
    header_row = None
    for i in range(min(20, len(raw))):
        if "コード" in str(raw.iat[i, 1]):
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame(columns=cols)
    data = raw.iloc[header_row + 1:, [0, 1, 2]].copy()
    data.columns = cols
    data["code"] = data["code"].map(_fmt_code)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "code"])
    data = data[data["code"] != ""]
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    data["name"] = data["name"].astype(str).str.strip()
    return data[cols]


def fetch_earnings_schedule(url: str, timeout: int = 60,
                            getter: Optional[Getter] = None, log=print) -> pd.DataFrame:
    """決算発表予定日ページに掲載されている Excel を全て取得・結合する（T-104）。

    ページには直近1〜2ヶ月分のリンクしか載らない（JPX 側のローリング更新。仕様であり
    不具合ではない）。G0ゲート（保有上限15営業日以内の決算除外）には実用上十分だが、
    100%網羅ではない。詳細は docs/DATA_SOURCES.md。リンクの URL は都度ページから抽出する
    （ハッシュ部分が変わるためハードコードしない）。取得失敗は例外を投げず、警告ログを
    出して取得できた分だけ返す（全滅なら空 DataFrame）。
    戻り値: DataFrame[code, date, name]
    """
    get = getter or _http_get_bytes
    cols = ["code", "date", "name"]
    try:
        html = get(url, timeout).decode("utf-8", errors="replace")
    except Exception as e:
        log(f"[earnings] ページ取得失敗: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=cols)

    hrefs = sorted(set(_XLSX_HREF_RE.findall(html)))
    if not hrefs:
        log("[earnings] Excel リンクが見つからない（ページ書式が変わった可能性）")
        return pd.DataFrame(columns=cols)

    frames = []
    for href in hrefs:
        full = _abs_url(href)
        try:
            content = get(full, timeout)
            t = _earnings_table_from_xlsx(content)
            if len(t):
                frames.append(t)
        except Exception as e:
            log(f"[earnings] {full} 取得失敗: {type(e).__name__}: {e} → スキップ")

    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["code", "date"]).sort_values(["date", "code"]).reset_index(drop=True)
    return df[["code", "date", "name"]]


def load_earnings_schedule(path: Path) -> pd.DataFrame:
    """reference/earnings_schedule.csv (code,date[,name]) → DataFrame[ticker,date]。無ければ空。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "date"])
    df = pd.read_csv(p, dtype=str).fillna("")
    df["ticker"] = df["code"].map(norm_ticker)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date"])[["ticker", "date"]].drop_duplicates()


def load_delistings(path: Path) -> pd.DataFrame:
    """reference/delistings.csv (code,name,date,reason) → DataFrame。無ければ空。"""
    p = Path(path)
    cols = ["ticker", "name", "date", "reason"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p, dtype=str).fillna("")
    df["ticker"] = df["code"].map(norm_ticker)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df[cols]


def load_manual_exclusions(path: Path, asof: pd.Timestamp) -> list[str]:
    """reference/manual_exclusions.csv (ticker,until,reason) を読み、asof 時点で
    有効な（until >= asof の）除外銘柄一覧を返す。無ければ空リスト。

    adjust.py の suspected_unrecorded_split（自動検出）を補う、既知のデータ品質
    問題の手動除外リスト（2026-08-27 追加）。例: 分割の効力発生前に yfinance が
    価格を先出し適用したが、fetch_ohlcv の取得窓（HISTORY_DAYS）が発生源の日付を
    含まないため check_splits の連続バー比較では検出できない銘柄。until を過ぎたら
    自動的に除外を終了する（手動でファイルを編集し続けなくてよい）。
    """
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p, dtype=str).fillna("")
    if len(df) == 0:
        return []
    df["ticker"] = df["ticker"].map(norm_ticker)
    df["until"] = pd.to_datetime(df["until"], errors="coerce").dt.normalize()
    asof = pd.Timestamp(asof)
    if asof.tz is not None:
        asof = asof.tz_localize(None)
    asof = asof.normalize()
    active = df[df["until"].notna() & (df["until"] >= asof)]
    return sorted(active["ticker"].unique().tolist())


def next_earnings_days(schedule: pd.DataFrame, asof: pd.Timestamp, ticker: str) -> Optional[int]:
    """asof 以降で最も近い決算発表日までの暦日数。無ければ None。"""
    if schedule is None or len(schedule) == 0:
        return None
    s = schedule[(schedule["ticker"] == ticker) & (schedule["date"] >= asof)]
    if len(s) == 0:
        return None
    return int((s["date"].min() - asof).days)


def next_earnings_business_days(schedule: pd.DataFrame, asof: pd.Timestamp,
                                ticker: str) -> Optional[int]:
    """asof 以降で最も近い決算発表日までの営業日数。無ければ None。

    土日のみを除外する近似（np.busday_count。祝日カレンダーは考慮しない）。
    D8 earnings_days（DESIGN.md §5）・G0（§2、保有上限 label_n 営業日）で共用する。
    """
    if schedule is None or len(schedule) == 0:
        return None
    s = schedule[(schedule["ticker"] == ticker) & (schedule["date"] >= asof)]
    if len(s) == 0:
        return None
    next_date = s["date"].min()
    return int(np.busday_count(np.datetime64(pd.Timestamp(asof), "D"),
                               np.datetime64(pd.Timestamp(next_date), "D")))
