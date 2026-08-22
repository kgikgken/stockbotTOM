# reference/

手動またはスクリプトで維持する参照データ。

- `listed_latest.csv` … JPX 上場銘柄一覧（自動更新・月曜/初回）
- `earnings_schedule.csv` … 決算発表予定日。列: code,date[,name]。ライブの決算除外に使う（M2 で自動取得を検討）
- `delistings.csv` … 上場廃止銘柄。列: code,name,date,reason。生存バイアスの上限評価に使う
