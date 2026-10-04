# theme-screener (preparation)

Japanese-equity screener, theme-based. It lives in the `theme-screener/`
directory of the stockbotTOM repository (placed here with the owner's
permission, 2026-10-04). It shares no code, data, tests or workflow with
stockbotTOM (chart-pattern version): it imports nothing from `src/stockbot/`,
writes only under this directory, and has no GitHub Actions workflow. Run
every command below from this directory.

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
universe/semicon_frontend.csv           listed companies (code is a string)
universe/semicon_frontend_boundary.csv  companies left out, with reason
src/themescreen/metrics.py              pure computations (no network)
src/themescreen/fetch.py                yfinance -> data/raw/<asof>/
src/themescreen/build_map.py            data/raw/<asof>/ -> out/
data/raw/<asof>/info.json, history.csv  raw inputs as fetched
out/semicon_frontend_<asof>.csv / .md   per-company values and report tables
tests/
```

## Run

```
python -m venv .venv && .venv/bin/pip install pandas numpy yfinance
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m src.themescreen.fetch --asof 2026-10-02
.venv/bin/python -m src.themescreen.build_map --asof 2026-10-02
```
