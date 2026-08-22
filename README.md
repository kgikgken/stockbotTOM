# stockbotTOM v8 — 日本株 順張り押し目スクリーナー

仕様は `docs/SPEC.md`、調査は `docs/RESEARCH.md`。本リポジトリはモジュール単位で段階的に構築する。

| モジュール | 内容 | 状態 |
|---|---|---|
| M1 | データ基盤（取得・整合性検査・保存・スナップショット・ユニバース） | 完了 |
| M2 | 特徴量（トレンド・押し目・出来高・収縮・相対力・週足） | 未着手 |
| M3 | スコアリング | 未着手 |
| M4 | 検証 L1 / L2 | 未着手 |
| M5 | 描画（既存デザイン踏襲）・LINE 配信 | 未着手 |

## セットアップ

```bash
python -m pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -t .        # ネットワーク不要
```

## 実行

```bash
SCREEN_DRYRUN=1 python -m stockbot.cli daily      # 合成データで全工程を通す
python -m stockbot.cli daily                      # 本番: JPX 一覧 → yfinance → 保存 → ユニバース
python -m stockbot.cli listed | fetch | universe  # 個別ステップ
```

## 生成物

```
data/
  reference/listed_latest.csv      上場銘柄一覧（JPX、失敗時は universe_jpx.csv を種に）
  store/ohlcv.csv.gz|parquet       日足の縦持ち（git には入れない。Actions cache で引き継ぐ）
  store/revisions.csv.gz           後から書き換わった足の記録
  store/split_issues.csv           分割の未調整（修正済）/ 未記録疑い（除外）
  store/fetch_meta.json            取得の内訳
  daily/YYYY-MM-DD.csv.gz          その日に新規観測した足（時点復元用。git にコミット）
  universe/latest.csv              ユニバース（採用フラグ付き）
  universe/universe_YYYY-MM-DD.csv 日付別
  universe/summary.json            件数内訳
```

## 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `DATA_DIR` | data | 生成物のルート |
| `SCREEN_DRYRUN` | 0 | 1 で合成データ |
| `FETCH_SCOPE` | auto | all / universe / auto（auto: 月曜と初回は全件、他は前回通過銘柄＋境界付近＋新規上場） |
| `HISTORY_DAYS` | 400 | 取得する履歴日数 |
| `FETCH_DEADLINE_SEC` | 5400 | 取得に使う上限秒数 |
| `MARKET_CLOSE_HHMM` | 15:30 | 引け時刻。これ以前の実行では当日足を捨てる |
| `MIN_ADV_JPY` | 2e8 | 20日平均売買代金の下限 |
| `MIN_PRICE` | 200 | 株価下限 |
| `MIN_HISTORY_BARS` | 250 | 採点に必要な履歴本数 |
| `ADV_WINDOW` | 20 | 売買代金の平均日数 |
| `MAX_STALENESS_DAYS` | 7 | 最終足がこれより古い銘柄を除外 |
| `REV_CLOSE_TOL` / `REV_VOLUME_TOL` | 0.005 / 0.05 | 改訂とみなす相対差 |
| `JPX_LISTED_URL` | JPX data_j.xls | 上場銘柄一覧の URL |
| `UNIVERSE_SEED_CSV` | universe_jpx.csv | JPX 取得失敗時の種 |

## 旧 v7.1 からの移行

1. `git tag v7.1-final && git push origin v7.1-final`
2. 旧ファイルを削除（`v7/`, `main_v7.py`, `backtest.py`, `grid_test.py`, `hypothesis_test.py`, `verify_level1.py`, `histogram.py`, `phase2_histogram.py`, `out_v7/`, `.github/workflows/v7*.yml`）
3. 残す: `src/worker.js`, `wrangler.toml`, `hero.jpeg`, `mascot.png`, `universe_jpx.csv`（M5 で `src/stockbot/render/assets/` と `notify/` に移設）
4. 本モジュールのファイルを配置して push。`data-daily` ワークフローを手動実行（`dryrun=1` で経路確認 → `dryrun=0`）
