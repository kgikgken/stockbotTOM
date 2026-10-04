"""EDINET API v2 client: find the latest annual securities report (有価証券報告書)
per company and download its XBRL (type=1) and XBRL-to-CSV (type=5) archives.

The API key is read from the environment variable EDINET_API_KEY and is never
printed or written to disk.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.edinet-fsa.go.jp/api/v2"
ANNUAL_REPORT = "120"  # 有価証券報告書
DOC_TYPE_XBRL = 1
DOC_TYPE_CSV = 5
EDINET_DIR = ROOT / "data" / "edinet"
SLEEP_SECONDS = 0.5


class EdinetError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("EDINET_API_KEY", "")
    if not key:
        raise EdinetError("EDINET_API_KEY is not set")
    return key


def check_api_error(body: bytes) -> None:
    """EDINET returns HTTP 200 with a JSON StatusCode on errors (e.g. 401)."""
    head = body[:200].lstrip()
    if not head.startswith(b"{"):
        return
    try:
        obj = json.loads(body)
    except ValueError:
        return
    status = obj.get("StatusCode", obj.get("metadata", {}).get("status"))
    if status is not None and str(status) != "200":
        msg = obj.get("message", obj.get("metadata", {}).get("message", ""))
        raise EdinetError(f"EDINET status {status}: {msg}")


def _get(path: str, params: dict[str, str | int], key: str) -> bytes:
    q = urllib.parse.urlencode({**params, "Subscription-Key": key})
    with urllib.request.urlopen(f"{API}/{path}?{q}", timeout=120) as r:
        body = r.read()
    check_api_error(body)
    return body


def list_documents(day: date, key: str, cache_dir: Path) -> list[dict]:
    """Document list of one submission date (type=2: with metadata). Cached."""
    cache = cache_dir / f"{day.isoformat()}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    body = _get("documents.json", {"date": day.isoformat(), "type": 2}, key)
    results = json.loads(body).get("results", [])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    time.sleep(SLEEP_SECONDS)
    return results


def pick_latest_annual(docs: list[dict], edinet_codes: set[str]) -> dict[str, dict]:
    """Latest annual report per EDINET code (by submitDateTime), not withdrawn."""
    best: dict[str, dict] = {}
    for d in docs:
        if d.get("docTypeCode") != ANNUAL_REPORT or d.get("edinetCode") not in edinet_codes:
            continue
        if str(d.get("withdrawalStatus", "0")) != "0":
            continue
        cur = best.get(d["edinetCode"])
        if cur is None or (d.get("submitDateTime") or "") > (cur.get("submitDateTime") or ""):
            best[d["edinetCode"]] = d
    return best


def download(doc_id: str, kind: int, key: str, out_dir: Path) -> Path:
    path = out_dir / f"{doc_id}_type{kind}.zip"
    if path.exists():
        return path
    body = _get(f"documents/{doc_id}", {"type": kind}, key)
    if not body.startswith(b"PK"):
        raise EdinetError(f"{doc_id} type={kind}: not a zip ({body[:80]!r})")
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    time.sleep(SLEEP_SECONDS)
    return path


def read_xbrl_csv(zip_path: Path) -> pd.DataFrame:
    """All XBRL-to-CSV files in a type=5 archive (UTF-16, tab-separated)."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            text = z.read(name).decode("utf-16")
            df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
            df.insert(0, "file", name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first submission date to scan")
    ap.add_argument("--end", required=True, help="last submission date to scan")
    args = ap.parse_args()
    key = api_key()
    with open(ROOT / "universe" / "edinet_codes.csv", encoding="utf-8") as f:
        companies = list(csv.DictReader(f))
    codes = {c["edinet_code"] for c in companies}
    docs: list[dict] = []
    day, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    while day <= end:
        docs.extend(list_documents(day, key, EDINET_DIR / "lists"))
        day += timedelta(days=1)
    latest = pick_latest_annual(docs, codes)
    rows: list[dict] = []
    for c in companies:
        d = latest.get(c["edinet_code"])
        row = {"code": c["code"], "edinet_code": c["edinet_code"], "doc_id": "", "period_end": "",
               "submitted": "", "xbrl_flag": "", "csv_flag": "", "download_seconds": "", "error": ""}
        if d is None:
            row["error"] = "annual report not found in scanned range"
        else:
            row.update(doc_id=d["docID"], period_end=d.get("periodEnd") or "",
                       submitted=d.get("submitDateTime") or "",
                       xbrl_flag=d.get("xbrlFlag") or "", csv_flag=d.get("csvFlag") or "")
            t0 = time.perf_counter()
            try:
                download(d["docID"], DOC_TYPE_CSV, key, EDINET_DIR / "docs")
                download(d["docID"], DOC_TYPE_XBRL, key, EDINET_DIR / "docs")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            row["download_seconds"] = f"{time.perf_counter() - t0:.2f}"
        rows.append(row)
        print(row)
    with open(EDINET_DIR / "annual_reports.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
