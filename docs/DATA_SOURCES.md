# DATA_SOURCES.md — 参照データの取得経路（T-104 調査結果）

対応: TASKS.md T-104。決算発表予定日・上場廃止銘柄一覧の自動取得可否を調査した結果と、
その制約をまとめる。取得ロジックは `src/stockbot/data/jpx_lists.py` の
`fetch_earnings_schedule()` / `fetch_delistings()`。

## 1. 決算発表予定日

- ページ: https://www.jpx.co.jp/listing/event-schedules/financial-announcement/index.html
- ページ内に「◯年◯月に四半期末又は期末を迎えた会社」という見出し行が複数あり、各行に
  Excel（.xlsx）へのリンクがある（例:
  `.../tvdivq0000001ofb-att/kessan07_0821.xlsx`）。URL のハッシュ部分
  （`tvdivq0000001ofb` 等）は更新のたびに変わるため、`fetch_earnings_schedule()` は
  毎回ページ本文から `href="....xlsx"` を正規表現で抽出する（ハードコードしない）。
- 見つかったリンクを**全てダウンロードして結合**する。

### 制約: ローリング更新で完全網羅ではない
- 実際に取得したページには直近2ヶ月分（例: 2026年6月分・7月分）の2ファイルしか
  掲載されていなかった。これは JPX 側の仕様であり、こちら側の不具合ではない。
- G0ゲート（保有上限15営業日以内の決算を除外する用途、DESIGN.md 参照）には実用上
  十分だが、「向こう1年分の決算日を全銘柄について把握する」といった網羅的な用途には
  使えない。過去分・数ヶ月先の分は掲載されないため取得できない。
- 毎営業日 `references` ステップを実行し、ローリングウィンドウを追従することで
  実用上の網羅性を確保する運用とする。

### xlsx の列構成（実データで確認済み）
1シート（シート名 `List`）、先頭3〜4行がタイトル・時点表記、5行目（0-indexed で4行目）が
見出し行、以降がデータ行。

| 列位置(0-indexed) | 見出し（日本語/英語） | 内容 |
|---|---|---|
| 0 | 決算発表予定日 / Scheduled Dates for Earnings Announcements | 発表予定日（日時） |
| 1 | コード / Code | 証券コード |
| 2 | 会社名 | 銘柄名（日本語） |
| 3 | Issue Name | 銘柄名（英語） |
| 4 | 決算期末 / Fiscal Year-end | 決算期末日 |
| 5〜6 | 業種名 / Industry | 業種 |
| 7〜8 | 種別 / Fiscal Year/Quarter | 四半期区分 |
| 9〜10 | 市場区分 / Market Segment | 市場区分 |

`fetch_earnings_schedule()` は列0(date)・列1(code)・列2(name) だけを使う
（`load_earnings_schedule()` が読む reference/earnings_schedule.csv の列と揃える）。
見出し行の位置をタイトル行数に依存させないよう、列1のセルに「コード」という文字列が
現れる行を探して見出し行と判定している（将来レイアウトが多少変わっても追従できるように
するため。位置を固定インデックスにはしていない）。

## 2. 上場廃止銘柄一覧

- ページ: https://www.jpx.co.jp/listing/stocks/delisted/index.html （当年分）
- HTML の `<table>` そのもの。列: 上場廃止日／銘柄名／コード／市場区分／上場廃止理由
- ページ内に「バックナンバー：2025年 2024年 … 2017年」という `<select class="backnumber">`
  があり、各 `<option>` の `value` 属性にバックナンバーページの href
  （`archives-01.html`〜`archives-09.html`。命名は確認できたが、`fetch_delistings()` は
  ハードコードせず毎回ページから抽出している）が入っている。
- 実データ確認時点で当年（2026年）+ 過去9年分（2025〜2017年）が選択肢にあり、
  ページ本文にも「過去11年以前の上場廃止銘柄につきましてはこちらでは掲載しておりません」
  と明記されている。`fetch_delistings(back_years=11)` の既定値はこれに合わせた上限。

### 制約
- 過去11年より前の上場廃止銘柄は JPX のこのページからは取得できない
  （生存バイアスの上限評価はこの期間に限られる。SPEC §4 緩和策1参照）。
- 上場廃止理由（`reason` 列）は正規化せず、ページの原文をそのまま保存する。
  観測された値の例:
  「（親会社名）の完全子会社化」「ＭＢＯ」「支配株主等による買収」「他社による買収」
  「上場維持基準への不適合」「内部管理体制等の改善が見込まれない」
  「申請による上場廃止」「会社合併」など。表記ゆれ（全角/半角等）がある。

## 3. 実装

- `fetch_delistings(url, back_years=11)` → `DataFrame[code, name, date, reason]`
- `fetch_earnings_schedule(url)` → `DataFrame[code, date, name]`
- どちらもページ構造の変化・ネットワーク障害時は例外を投げず、警告ログを出して
  取得できた分だけ返す（`load_listed_with_fallback` と同じ考え方）。取得0件の場合、
  呼び出し側（`cli.py` の `step_references`）は reference/ の既存ファイルを上書きしない。
- `python -m stockbot.cli references` で `data/reference/earnings_schedule.csv` /
  `data/reference/delistings.csv` を更新する。`SCREEN_DRYRUN=1` ではネットワークを
  叩かず何もしない（DRYRUN で参照データの新規性は検証しない）。

## 4. 指数データ（D2相対力・地合いゲージ §8.1 用、TOPIX/日経225代替）

対応: `src/stockbot/data/yf_fetch.py` の `fetch_index()` / `INDEX_CANDIDATES`。
DESIGN.md §12「IDX は TOPIX、無ければ日経225」の実装経路を記録する。

### 候補チェーン（2026-08-24 に実インターネット環境で確認）

`("^TPX", "1306.T", "^N225")` を順に試す。

| ティッカー | 内容 | 結果 |
|---|---|---|
| `^TPX` | TOPIX（生指数） | yfinance 上でほぼ常に取得不能（"possibly delisted; no price data found"）。`^TOPX` も同様に不可 |
| `1306.T` | NEXT FUNDS TOPIX連動型上場投信（野村アセットマネジメント、2001年上場） | 取得可能。TOPIXを忠実に追跡する。2016-02〜2026-08の期間で分割イベント0件を確認済み（`scripts/probe_index_tickers.py` で確認、分割リスクなし） |
| `^N225` | 日経225 | 取得可能。TOPIXとは構成銘柄・加重方式が異なる別の指数。1306.Tも取得できない場合の最後の手段 |

`1348.T`（MAXISトピックス上場投信）・`1475.T`（iシェアーズ・コアTOPIX ETF）も同様に取得可能・分割イベント0件を確認済みだが、1306.Tが最も上場が古く流動性が高いため第一候補にした。

`cli.py step_index` が実際に使ったティッカーを `data/store/index_meta.json` に記録し
（`{"ticker", "label", "is_fallback"}`）、`validation.report`（T-405）が L1レポート冒頭に
明記する。`is_fallback` は日経225の場合のみ True（1306.TはTOPIX扱いのままフォールバック
警告を出さない、TOPIXを追跡する別物ではないため）。

### 既知の制約: 1306.Tの分配金による小さな時点ノイズ

`fetch_index`/`fetch_ohlcv` はいずれも `auto_adjust=False` で取得する（配当・分配未調整、
SPEC §4）。個別銘柄の配当落ちと生のTOPIXの配当落ちは同じ仕組み（未調整の生値）なので
対称的だが、1306.Tは投資信託として分配金を**年1〜2回まとめて**分配するため、その
分配落ち日には1306.Tの価格だけが一時的に下振れする（生のTOPIXは個別銘柄の配当落ちが
年間を通じて分散的に反映されるため、同じタイミングでの単発的な下落は生じない）。

結果として、分配落ち日をまたぐ60日/120日窓（`d2_rs60`/`d2_rs120`、DESIGN.md §5）では
指数側だけが一時的に下振れし、その窓で計算される**全銘柄のD2が一律わずかに上振れ**する。
同一日内の銘柄間の順位付け（スコアリングの横断面比較）には影響しないが、異なる期間を
またぐ比較（L1のIC・分位曲線などの期間集計、DESIGN.md §10.2）には小さなノイズ源になり
うる。分配日の実績は `data/store` の `__IDX__` 系列（`Dividends` 列）から確認できる。
2026-08-26 の窓別カバレッジ確認では、1306.Tの単日下落（対数リターン<-2%、配当落ち含む）
が主評価窓で50日、頑健性窓で49日と近い頻度で観測されており、窓間で非対称な影響は
無いと判断している（診断のみ・修正はしない。`scripts/compute_robustness_start.py`）。

### 指数データの欠損と前方補完（2026-08-26 追加）

`1306.T`（および`^N225`フォールバック）は、個別株の営業日集合と完全には一致しない
散発的な1日程度の取得欠落が稀にある。2026-08-26 の実測では、主評価窓
（2021-08-01〜2026-01-30）は欠落0件（0.00%）だったのに対し、頑健性窓
（2017-03-15〜2021-07-31）は21日（1.93%、最大連続3営業日=2018年始）の欠落があった
（`scripts/compute_robustness_start.py`、`probe-robustness-start.yml`）。

`pipeline.compute_daily_features` は `idx_close.reindex(close.index)` で銘柄の日付に
指数を整列した後、**最大3営業日まで前方補完**する（`.ffill(limit=3)`）。ffillは定義上
過去の値のみを使うため未来参照にはならない。3営業日を超える欠損は補完せずNaNのまま
残す。これを入れないと、欠損した日のD2（d2_rs60/d2_rs120/d2_rsline_pos）がDESIGN.md
§6.1の規約により「判定不能→F10〜F12通過扱い」になり、指数データの欠落率が異なる窓
どうしで異なるフィルタを測ることになる（頑健性窓の方が主評価窓よりF10〜F12が緩く
なる）。主評価窓は欠落が0件のためこの変更は実質的に無効（値は変わらない）で、頑健性窓
にのみ実効的に効く。
