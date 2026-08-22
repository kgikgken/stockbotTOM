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
