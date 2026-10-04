import tempfile
import unittest
import zipfile
from pathlib import Path

from .. import _path  # noqa: F401
from themescreen.edinet import EdinetError, check_api_error, pick_latest_annual, read_xbrl_csv


class TestCheckApiError(unittest.TestCase):
    def test_401_in_200_body(self) -> None:
        with self.assertRaises(EdinetError):
            check_api_error(b'{"StatusCode": 401,"message": "Access denied"}')

    def test_metadata_error(self) -> None:
        with self.assertRaises(EdinetError):
            check_api_error(b'{"metadata": {"status": "404", "message": "Not Found"}}')

    def test_ok(self) -> None:
        check_api_error(b'{"metadata": {"status": "200"}, "results": []}')
        check_api_error(b"PK\x03\x04binary")


class TestPickLatestAnnual(unittest.TestCase):
    def test_latest_and_filters(self) -> None:
        docs = [
            {"docID": "A", "edinetCode": "E1", "docTypeCode": "120", "submitDateTime": "2025-06-20 15:00", "withdrawalStatus": "0"},
            {"docID": "B", "edinetCode": "E1", "docTypeCode": "120", "submitDateTime": "2026-06-19 15:00", "withdrawalStatus": "0"},
            {"docID": "C", "edinetCode": "E1", "docTypeCode": "130", "submitDateTime": "2026-07-01 15:00", "withdrawalStatus": "0"},
            {"docID": "D", "edinetCode": "E1", "docTypeCode": "120", "submitDateTime": "2026-08-01 15:00", "withdrawalStatus": "1"},
            {"docID": "E", "edinetCode": "E9", "docTypeCode": "120", "submitDateTime": "2026-06-19 15:00", "withdrawalStatus": "0"},
        ]
        r = pick_latest_annual(docs, {"E1"})
        self.assertEqual(list(r), ["E1"])
        self.assertEqual(r["E1"]["docID"], "B")


class TestReadXbrlCsv(unittest.TestCase):
    def test_utf16_tsv(self) -> None:
        tsv = "要素ID\t項目名\t値\njpcrp_cor:NetSales\t売上高\t100\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("XBRL_TO_CSV/a.csv", tsv.encode("utf-16"))
                z.writestr("XBRL_TO_CSV/readme.txt", b"x")
            df = read_xbrl_csv(p)
        self.assertEqual(df["要素ID"].tolist(), ["jpcrp_cor:NetSales"])
        self.assertEqual(df["値"].tolist(), ["100"])
        self.assertEqual(df["file"].tolist(), ["XBRL_TO_CSV/a.csv"])


if __name__ == "__main__":
    unittest.main()
