# CLAUDE.md — 作業規約（Claude Code / 実装者向け）

このリポジトリは日本株のスイングトレード向け「順張り押し目」スクリーナー（stockbotTOM v8）。
毎営業日 7:00 JST に GitHub Actions で動き、候補をランキングして LINE に画像で配信する。

## 正の情報源（この順で優先）

**19 条件のスクリーナーは 2026-09-08 に撤去した**（`docs/SCREENER_CLOSING.md`）。
現在の作業対象は 7 パターンの検出（`docs/PATTERN.md`）。判定式は 2026-09-08 に揃い
（§4 Q-1 クローズ）、7 つとも実装済み。**検出数を出すだけで、記録も配信もまだしない。**
検証プロジェクト（v8 順張り押し目）は 2026-08-30 に終了している（`docs/CLOSING.md`）。

1. `docs/PATTERN.md` — **現在の作業対象。** 7 パターンの仕様と事前登録パラメータ。
   質問ログもここ（§4）
2. `docs/SCREENER_CLOSING.md` — 撤去の範囲・保全した記録・未達だった判定。いま生きて
   いるコードの範囲はここに書いてある
3. `docs/SPEC.md` — 決定事項。ここに無いことは決まっていない
4. `docs/RESEARCH.md` — 根拠。設計を変える根拠にはしない（変えるのは設計責任者）

終了済みで**編集しない**もの: `docs/DESIGN.md`、`docs/TASKS.md`、`docs/CLOSING.md`、
`docs/FUTURE_HYPOTHESIS.md`、`docs/SCREENER.md`。DESIGN.md と SCREENER.md はコードの式や
記録の列が何かを引くために読んでよいが、そこの条件・閾値・検定数・停止規則・撤退基準を
新しい作業に引き継がない。

**パターンの判定式をこちらで作らない。** 判定式は PATTERN.md §2 にあるものが全部で、
書いていない条件を足さない。事前登録パラメータ（§1）は決まっているので動かさない。
§2.2 の「実装上の読み」は判定式の追加ではなく、既にある数値をどこに当てるかの記録である。

チャットの指示と文書が食い違ったら、文書を正として指摘する。

## 作業の進め方
- 実装前に受け入れ条件と必須テストを読み、テストを先に書く
- PATTERN.md に書いていない判断が必要になったら、実装せず PATTERN.md の質問ログ（§4）に追記して止まる。推測で埋めない
- 完了報告には次を含める: 変更ファイル一覧、テスト結果（件数）、PATTERN.md のどの節を実装したか、設計責任者のレビュー対象箇所（スイング確定ラグ・時点整合に触れた場合は必ず明記）

## 絶対に守ること
- **未来参照の禁止**: すべての量は T の引けまでのデータで計算する。スイングは確定ラグ k 本後にしか使えない。週足は T を含む週を使わない。ラベルだけが T+1 以降を見る
- **再計算一致テスト**（DESIGN.md §11）を新しい特徴量・指標すべてに適用する。これが落ちたら他が通っていてもマージしない
- **パラメータを増やさない**: 新しい閾値が欲しければ質問ログへ
- **閾値を勝手に動かさない**: PATTERN.md §1 の事前登録パラメータ（ε・旗竿・タッチ点数・±1.5%）は検出数を見てから動かさない。新しいものを推測で作らない
- **スコアを作らない**: スコア計算・F 除外・プール百分位を使わない（SCREENER.md §2.2 の方針を引き継ぐ）
- **±1.5% を ATR 連動にしない**: 銘柄のボラティリティで許容幅が変わると、同じパターン名が銘柄ごとに別の形を指す（PATTERN.md §1）
- **ホールドアウト（2026-02〜2026-08）を見ない**: 検証 L1 や設計途中で参照するコードを書かない。`validation/replay.py` はホールドアウト生成を明示フラグなしで行わない
- **LINE 経路を変えない**: `src/worker.js`、`wrangler.toml`、Secrets 名（`LINE_CHANNEL_ACCESS_TOKEN` / `LINE_TO` / `WORKER_URL` / `WORKER_AUTH_TOKEN`）
- **保存データをコミットしない**: `data/store/` は `.gitignore`。`data/daily/`、`data/universe/`、`data/reference/` はコミットする
- **配信記録を消さない**: `data/daily/` の `delivered_*.csv` / `screen_summary_*.json` / `outcome_*.csv` は撤去後も保全する（SCREENER_CLOSING.md）
- **配信を勝手に再開しない**: ワークフローの Notify ステップは外してある。戻すのは設計責任者の指示があったときだけ
- **検証結果を解釈しない**: 表と図を出すまで。採否・継続・撤退の判断は設計責任者

## 環境と規約
- Python 3.12、pandas 3.x（Copy-on-Write 既定。連鎖代入をしない。`df.loc[...] = ...` か `assign` を使う）、numpy
- テストは `unittest` 互換、ネットワーク不要。`python -m unittest discover -s tests -t .`
- `SCREEN_DRYRUN=1` で合成データにより全工程が通ること。新しい段を足したら DRYRUN 経路も足す
- yfinance は関数内で遅延 import（テストと DRYRUN で不要）
- ログは `print`。日本語可。絵文字は使わない
- 型ヒント必須。docstring に SCREENER.md の節番号を書く（既存の検証コードは DESIGN.md の節番号のまま）
- ファイル配置
  ```
  src/stockbot/
    config.py  cli.py  pipeline.py
    screener/    record.py  resolver.py          (conditions.py / screen.py は撤去済み)
    notify/      message.py  line_send.py
    data/        yf_fetch.py  adjust.py  store.py  jpx_lists.py  synthetic.py
    universe/    build.py
    features/    indicators.py  swings.py  pullback.py  dimensions.py  regime.py  sector.py
                 pattern.py（7 パターン。PATTERN.md §2.1・§2.2）
    scoring/     composite.py  template.py  ranking.py
    validation/  labels.py  replay.py  layer1.py  report.py  calibration.py
    render/      context.py  template.html  render.py
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
