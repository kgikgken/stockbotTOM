"""JPX 参照データ取得（T-104）。保存済み HTML/xlsx フィクスチャのみを使い、
ネットワークには依存しない（getter を差し替えて注入する）。"""
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import _path  # noqa: F401
from stockbot.data.jpx_lists import (
    _delisted_table_from_html,
    _earnings_table_from_xlsx,
    _fmt_code,
    fetch_delistings,
    fetch_earnings_schedule,
    load_manual_exclusions,
)

# ------------------------------------------------------------------ fixtures
def _delisted_html(rows: list[tuple[str, str, str, str, str]], backnumbers: list[tuple[str, str]] = ()) -> str:
    """実ページの構造を模した最小 HTML（上場廃止日/銘柄名/コード/市場区分/上場廃止理由）。"""
    trs = "".join(
        f"<tr><td>{d}</td><td>{n}</td><td>{c}</td><td>{m}</td><td>{r}</td></tr>"
        for d, n, c, m, r in rows
    )
    options = "".join(f'<option value="{href}">{year}年</option>' for href, year in backnumbers)
    return f"""
    <html><body>
    <div class="number-select"><select class="backnumber">{options}</select></div>
    <table>
      <thead><tr><th>上場廃止日</th><th>銘柄名</th><th>コード</th><th>市場区分</th><th>上場廃止理由</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>
    </body></html>
    """


def _earnings_html(hrefs: list[str]) -> str:
    links = "".join(
        f'<td><a href="{h}" rel="external"><img src="/common/images/icon/icon-xls.png"/></a></td>'
        for h in hrefs
    )
    return f"<html><body><table><tr>{links}</tr></table></body></html>"


def _earnings_xlsx_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    """実 xlsx の構造（タイトル3行 + 見出し行(コード列を含む) + データ行）を再現する。"""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "List"
    ws.append(["◯月に期末を迎えた会社の一覧"])
    ws.append(["List of companies..."])
    ws.append(["2026年8月20日 現在"])
    ws.append(["As of 2026/8/20"])
    ws.append(["決算発表予定日\nScheduled Dates", "コード\nCode", "会社名", "Issue Name"])
    for date, code, name in rows:
        ws.append([date, code, name, name])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class FmtCodeTest(unittest.TestCase):
    def test_int_like_float_has_no_decimal(self):
        self.assertEqual(_fmt_code(7590.0), "7590")

    def test_alnum_code_preserved(self):
        self.assertEqual(_fmt_code("436a"), "436A")

    def test_nan_becomes_empty(self):
        self.assertEqual(_fmt_code(float("nan")), "")


class DelistedTableParseTest(unittest.TestCase):
    def test_parses_expected_columns(self):
        html = _delisted_html([
            ("2026/05/01", "（株）テスト", "1234", "スタンダード", "申請による上場廃止"),
            ("2026/04/28", "テスト２（株）", "5678", "プライム", "ＭＢＯ"),
        ])
        df = _delisted_table_from_html(html)
        self.assertListEqual(list(df.columns), ["code", "name", "date", "reason"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["code"], "1234")
        self.assertEqual(df.iloc[0]["date"], "2026-05-01")
        self.assertEqual(df.iloc[1]["reason"], "ＭＢＯ")

    def test_no_matching_table_returns_empty(self):
        df = _delisted_table_from_html("<html><body><p>no table here</p></body></html>")
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ["code", "name", "date", "reason"])


class FetchDelistingsTest(unittest.TestCase):
    def test_follows_backnumber_links_and_merges(self):
        current_url = "https://example.test/delisted/index.html"
        archive_url = "https://example.test/delisted/archives-01.html"
        pages = {
            current_url: _delisted_html(
                [("2026/05/01", "現行銘柄", "1111", "プライム", "申請による上場廃止")],
                backnumbers=[(current_url, "2026"), (archive_url, "2025")],
            ),
            archive_url: _delisted_html(
                [("2025/03/01", "旧銘柄", "2222", "スタンダード", "ＭＢＯ")]
            ),
        }

        def getter(url, timeout):
            return pages[url].encode("utf-8")

        df = fetch_delistings(current_url, back_years=11, getter=getter, log=lambda m: None)
        self.assertEqual(set(df["code"]), {"1111", "2222"})
        self.assertListEqual(list(df.columns), ["code", "name", "date", "reason"])

    def test_back_years_limits_archive_pages_fetched(self):
        current_url = "https://example.test/delisted/index.html"
        calls = []

        def getter(url, timeout):
            calls.append(url)
            if url == current_url:
                return _delisted_html(
                    [("2026/05/01", "A", "1111", "プライム", "申請による上場廃止")],
                    backnumbers=[(current_url, "2026"), ("/a1.html", "2025"), ("/a2.html", "2024")],
                ).encode("utf-8")
            return _delisted_html([]).encode("utf-8")

        fetch_delistings(current_url, back_years=2, getter=getter, log=lambda m: None)
        # 当年 + バックナンバー1件(back_years-1=1) のみ取得しているはず
        self.assertEqual(len(calls), 2)

    def test_archive_failure_does_not_abort_whole_fetch(self):
        current_url = "https://example.test/delisted/index.html"

        def getter(url, timeout):
            if url == current_url:
                return _delisted_html(
                    [("2026/05/01", "A", "1111", "プライム", "申請による上場廃止")],
                    backnumbers=[(current_url, "2026"), ("/broken.html", "2025")],
                ).encode("utf-8")
            raise RuntimeError("network down")

        df = fetch_delistings(current_url, getter=getter, log=lambda m: None)
        self.assertEqual(list(df["code"]), ["1111"])

    def test_current_page_failure_returns_empty(self):
        def getter(url, timeout):
            raise RuntimeError("boom")

        df = fetch_delistings("https://example.test/delisted/index.html", getter=getter, log=lambda m: None)
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ["code", "name", "date", "reason"])


class EarningsXlsxParseTest(unittest.TestCase):
    def test_parses_data_rows_after_header(self):
        content = _earnings_xlsx_bytes([
            ("2026-08-25", "7590", "タカショー"),
            ("2026-08-27", "436A", "サイバーソリューションズ"),
        ])
        df = _earnings_table_from_xlsx(content)
        self.assertListEqual(list(df.columns), ["date", "code", "name"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["code"], "7590")
        self.assertEqual(df.iloc[0]["date"], "2026-08-25")
        self.assertEqual(df.iloc[1]["code"], "436A")

    def test_malformed_workbook_returns_empty(self):
        df = _earnings_table_from_xlsx(b"not an xlsx file")
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ["date", "code", "name"])


class FetchEarningsScheduleTest(unittest.TestCase):
    def test_extracts_and_merges_multiple_xlsx_links(self):
        page_url = "https://example.test/earnings/index.html"
        href1 = "/earnings/att1/kessan06.xlsx"
        href2 = "/earnings/att2/kessan07.xlsx"
        xlsx1 = _earnings_xlsx_bytes([("2026-08-10", "1000", "A社")])
        xlsx2 = _earnings_xlsx_bytes([("2026-08-25", "2000", "B社")])

        def getter(url, timeout):
            if url == page_url:
                return _earnings_html([href1, href2]).encode("utf-8")
            if url.endswith("kessan06.xlsx"):
                return xlsx1
            if url.endswith("kessan07.xlsx"):
                return xlsx2
            raise AssertionError(f"unexpected url {url}")

        df = fetch_earnings_schedule(page_url, getter=getter, log=lambda m: None)
        self.assertEqual(set(df["code"]), {"1000", "2000"})
        self.assertListEqual(list(df.columns), ["code", "date", "name"])

    def test_no_xlsx_links_returns_empty(self):
        page_url = "https://example.test/earnings/index.html"

        def getter(url, timeout):
            return b"<html><body>no links</body></html>"

        df = fetch_earnings_schedule(page_url, getter=getter, log=lambda m: None)
        self.assertEqual(len(df), 0)

    def test_page_failure_returns_empty(self):
        def getter(url, timeout):
            raise RuntimeError("boom")

        df = fetch_earnings_schedule("https://example.test/earnings/index.html",
                                     getter=getter, log=lambda m: None)
        self.assertEqual(len(df), 0)


class LoadManualExclusionsTest(unittest.TestCase):
    """データ品質による手動除外リスト（2026-08-27 追加、9900.T 先出し分割対応）。
    adjust.py の自動検出をすり抜けたケースを until 日付付きで一時的に除外する。"""

    def test_active_exclusion_within_until_date(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "manual_exclusions.csv"
            p.write_text("ticker,until,reason\n9900.T,2026-08-31,split pre-applied\n",
                         encoding="utf-8")
            result = load_manual_exclusions(p, pd.Timestamp("2026-08-27"))
            self.assertEqual(result, ["9900.T"])

    def test_exclusion_expires_after_until_date(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "manual_exclusions.csv"
            p.write_text("ticker,until,reason\n9900.T,2026-08-31,split pre-applied\n",
                         encoding="utf-8")
            result = load_manual_exclusions(p, pd.Timestamp("2026-09-01"))
            self.assertEqual(result, [], "untilを過ぎたら自動的に除外を終了するはず")

    def test_until_date_itself_is_still_active(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "manual_exclusions.csv"
            p.write_text("ticker,until,reason\n9900.T,2026-08-31,split pre-applied\n",
                         encoding="utf-8")
            result = load_manual_exclusions(p, pd.Timestamp("2026-08-31"))
            self.assertEqual(result, ["9900.T"])

    def test_missing_file_returns_empty(self):
        result = load_manual_exclusions(Path("/does/not/exist.csv"), pd.Timestamp("2026-08-27"))
        self.assertEqual(result, [])

    def test_ticker_normalized_to_dot_t_suffix(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "manual_exclusions.csv"
            p.write_text("ticker,until,reason\n9900,2026-08-31,split pre-applied\n",
                         encoding="utf-8")
            result = load_manual_exclusions(p, pd.Timestamp("2026-08-27"))
            self.assertEqual(result, ["9900.T"])


if __name__ == "__main__":
    unittest.main()
