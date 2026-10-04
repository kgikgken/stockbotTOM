# theme-screener (draft)

Status: draft. The design is not confirmed; the confirmed version will be
placed in `docs/theme/`.

Japanese-equity screener, theme-based (テーマ版). It lives in the stockbotTOM
repository next to the chart-pattern version (パターン版, `src/stockbot/`).
It imports nothing from `src/stockbot/`, writes only under `data/theme/`
(never `data/daily/`), and has no workflow.

Hypothesis tests of the theme version are counted separately from
`N_TESTS_TOTAL` of the pattern version (15, `src/stockbot/validation/pattern_strata.py`).
None has been run so far.

## Current task (as stated by the owner, 2026-10-03)

- Build an industry map for exactly one industry: semiconductor front-end
  process (半導体前工程). End-to-end preparation.
- Purpose: design validation. Measured: only whether each item can be computed.
- Not measured: performance (subsequent stock price). Nothing after asof is
  fetched, stored, or referenced.

Steps given by the owner:

1. List Japanese listed companies related to the front-end process, classified
   by process: 露光・成膜・エッチング・洗浄・検査・搬送・部材・ガス・薬液・ウェハ
2. Fetch per company (yfinance): market cap, 60-day median trading value,
   whether the latest fiscal period is an operating loss
3. Cutoffs: 60-day median trading value < 100M JPY -> excluded;
   latest-period operating loss -> excluded
4. Table by process. Columns: code (string), name, process, market cap,
   60-day median trading value

## Definitions (decided in this implementation; the owner may change them)

| Item | Definition |
|---|---|
| asof | Last completed session. First run: 2026-10-02 |
| Market cap | yfinance `info["marketCap"]` at fetch time, currency must be JPY |
| Trading value | `Close x Volume` of a daily bar, `history(auto_adjust=False)`. Both are split-adjusted by yfinance, so the product is unaffected by splits. An approximation of the exchange's 売買代金 |
| 60-day median | Median over the last 60 sessions on or before asof. Rows with NaN close/volume are dropped. Fewer than 60 -> not computable |
| Latest period | Latest column of yfinance annual `income_stmt`. If its value is missing it is not computable; no fallback to an older column |
| Operating loss | `Operating Income` < 0. Zero is not a loss. `Total Operating Income As Reported` is saved for comparison only |
| Cutoff status | 掲載: both inputs computable and no cutoff triggered. 除外: any computable cutoff triggered. 判定不能: nothing triggered and an input is missing |
| Process | One process per company (the process of its main front-end product, `basis`). Other processes are kept in `other_processes`. The assignment is a judgment and is listed in the report |
| 検査 | In-process inspection and metrology (mask, defect, CD). Wafer electrical test (tester, prober, probe card) is treated as back-end and not listed |
| Not in the 10 categories | CMP equipment, ultrapure water. Not listed; shown in `universe/semicon_frontend_boundary.csv` |
| Row order | Process order above, then code ascending. Not a ranking |

## Layout

```
src/themescreen/                  config.py metrics.py fetch.py build_map.py edinet.py
tests/themescreen/                test_metrics.py test_edinet.py
data/theme/universe/              semicon_frontend.csv (listed), semicon_frontend_boundary.csv (left out), edinet_codes.csv
data/theme/raw/<asof>/            info.json, history.csv (yfinance, as fetched)
data/theme/out/                   semicon_frontend_<asof>.csv / .md
data/theme/edinet/                lists/ (daily document lists), annual_reports.csv; docs/ is git-ignored
docs/theme/README.md              this file
```

## Run

```
python -m unittest discover -s tests -t .      # whole suite, including tests/themescreen
PYTHONPATH=src python -m themescreen.fetch --asof 2026-10-02
PYTHONPATH=src python -m themescreen.build_map --asof 2026-10-02
```
