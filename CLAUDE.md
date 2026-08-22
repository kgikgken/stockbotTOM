# CLAUDE.md — 作業規約（Claude Code / 実装者向け）

このリポジトリは日本株のスイングトレード向け「順張り押し目」スクリーナー（stockbotTOM v8）。
毎営業日 7:00 JST に GitHub Actions で動き、候補をランキングして LINE に画像で配信する。

## 正の情報源（この順で優先）
1. `docs/SPEC.md` — 決定事項。ここに無いことは決まっていない
2. `docs/DESIGN.md` — 詳細設計。式・閾値・手順はここに従う
3. `docs/TASKS.md` — 作業分割。受け入れ条件と必須テスト。質問ログもここ
4. `docs/RESEARCH.md` — 根拠。設計を変える根拠にはしない（変えるのは設計責任者）

チャットの指示と文書が食い違ったら、文書を正として指摘する。

## 作業の進め方
- 1 回の作業は TASKS.md の 1 タスク。タスク ID をブランチ名とコミットメッセージに入れる（例 `feat(T-203): pullback state machine`）
- 実装前に受け入れ条件と必須テストを読み、テストを先に書く
- DESIGN.md に書いていない判断が必要になったら、実装せず TASKS.md の質問ログに追記して止まる。推測で埋めない
- 完了報告には次を含める: 変更ファイル一覧、テスト結果（件数）、DESIGN.md のどの節を実装したか、設計責任者のレビュー対象箇所（スイング確定ラグ・ラベル・時点整合に触れた場合は必ず明記）

## 絶対に守ること
- **未来参照の禁止**: すべての量は T の引けまでのデータで計算する。スイングは確定ラグ k 本後にしか使えない。週足は T を含む週を使わない。ラベルだけが T+1 以降を見る
- **再計算一致テスト**（DESIGN.md §11）を新しい特徴量・指標すべてに適用する。これが落ちたら他が通っていてもマージしない
- **パラメータを増やさない**: DESIGN.md §12 にある値だけ。新しい閾値が欲しければ質問ログへ
- **重みを自由にしない**: スコアは V1/V2/V3 の 3 変種だけ
- **ホールドアウト（2026-02〜2026-08）を見ない**: 検証 L1 や設計途中で参照するコードを書かない。`validation/replay.py` はホールドアウト生成を明示フラグなしで行わない
- **LINE 経路を変えない**: `src/worker.js`、`wrangler.toml`、Secrets 名（`LINE_CHANNEL_ACCESS_TOKEN` / `LINE_TO` / `WORKER_URL` / `WORKER_AUTH_TOKEN`）
- **保存データをコミットしない**: `data/store/` は `.gitignore`。`data/daily/`、`data/universe/`、`data/reference/` はコミットする
- **検証結果を解釈しない**: 表と図を出すまで。採否・継続・撤退の判断は設計責任者

## 環境と規約
- Python 3.12、pandas 3.x（Copy-on-Write 既定。連鎖代入をしない。`df.loc[...] = ...` か `assign` を使う）、numpy
- テストは `unittest` 互換、ネットワーク不要。`python -m unittest discover -s tests -t .`
- `SCREEN_DRYRUN=1` で合成データにより全工程が通ること。新しい段を足したら DRYRUN 経路も足す
- yfinance は関数内で遅延 import（テストと DRYRUN で不要）
- ログは `print`。日本語可。絵文字は使わない
- 型ヒント必須。docstring に DESIGN.md の節番号を書く
- ファイル配置
  ```
  src/stockbot/
    config.py  cli.py  pipeline.py
    data/        yf_fetch.py  adjust.py  store.py  jpx_lists.py  synthetic.py
    universe/    build.py
    features/    indicators.py  swings.py  pullback.py  dimensions.py  regime.py
    scoring/     composite.py  template.py  ranking.py
    validation/  labels.py  replay.py  layer1.py  layer2.py  baselines.py  report.py  resolver.py
    render/      template.html  render.py  assets/
    notify/      line_send.py
  tests/
  docs/
  data/  (reference/ daily/ universe/ をコミット、store/ は除外)
  ```

## よくある落とし穴（旧版で実際に起きたもの）
- DRYRUN と本番が同じ `data/` を共有すると、合成データが git コミットや Actions cache 経由で本番データに混入する（2026-08-22 に実際に発生）。DRYRUN は既定で `data-dryrun/` に分離済み（`config.py`）。新しい書き込み先を追加するときは、DRYRUN と本番が物理的に分かれることを確認する
- ザラ場中に実行すると当日足が未確定のまま入り、再実行のたびに結果が変わる → `clean_frame` が引け前の当日足を落とす。引け後実行では含める
- 四本値の一部が NaN の行は比較が素通りして候補に混ざる → 取得時に落とす
- 単一銘柄取得でも MultiIndex 列が返る → `flatten_single`
- `git push || echo` で失敗を握り潰すと「更新されない」原因が分からない → 失敗は赤にする
- 日次で肥大するファイルを毎回丸ごとコミットするとリポジトリが膨らむ → 日付別ファイル

## 完了の定義
- 受け入れ条件を満たす
- 必須テストと既存テストが全件通る
- DRYRUN で `python -m stockbot.cli daily` が通る（該当段がある場合）
- 完了報告に上記の項目が揃っている
