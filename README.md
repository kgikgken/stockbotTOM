# stockbotTOM — v5.0 歪み×資金循環スクリーニング(パス1自動化)

歪み(反転・PEAD)を主エンジン、資金循環を文脈フィルタとする v5.0 枠組みの
**定量パス1** を自動化した構成です。旧トレンドフォロー(v2.0系)および v4.1 の
タイプ体系(A〜G)は廃止し、エンジン体系(A/S/B疑い)へ再編しています。
`utils/` 依存はありません(`mispricing/` に自己完結)。

## エンジン構成(v5.0)

| エンジン | 内容 | ボット(パス1) |
|---|---|---|
| **A(主力・ロング)** | 業種内リバーサル×非ファンダ性×クオリティ | 自動(業種指数比の相対乖離・押し目2〜5日) |
| **S(ショート)** | (b)逆流戻り売り(流出セクター成熟以降の2〜3日リバウンド) | 自動 / (a)悪材料イベント型はチャット側 |
| **B疑い(ロング)** | PEAD(決算月2/5/8/11のみ・上方イベント痕跡+ドリフト限定) | 痕跡検知のみ(上振れ確認はユーザー) |
| G(制度イベント) | 指数リバランス・TOB特殊局面 | チャット側 |

旧C(需給)・E(テクニカル)はエンジンAの補助トリガー(RSI・絶対乖離)に統合、
単独の点灯根拠にはしません。旧D(出遅れ追随ロング)は廃止のままです。

## ボットの担当範囲(v5.0仕様との対応)

ボットの「本命」は全件「仮点灯(未確認)」です。確定候補への昇格は、
iSPEED照合(独立2ソース化)とチャット側のゲート0(スティールマン)・
ゲート3(反証/プレモータム)を経てから行ってください。

- 自動: STEP1地合い / STEP1.5資金循環マップ(候補ゼロでも出力) / 流動性フィルタ /
  エンジンA・S・B疑いのトリガー / 非ファンダ性の代理判定 / R床・到達確率・
  概算ネットR・出口設計(円数字) / 保有ポジション評価 / STEP5ログ・棄却台帳
- チャット側(パス2): ゲート0・ゲート3 / ニュース2ソース照合 / クオリティ・キルの
  財務確認 / エンジンG / エンジンS(a)悪材料イベント型

単一ソース(yfinance)を許容し確信度は減点しませんが、本命は全件仮点灯です。

## 保有ポジション評価(新機能)

`positions.csv` の各銘柄を、毎回**新規候補と全く同じエンジン判定**に本日のデータで通します。

- 「今スクリーニングしたら何点か」: 継続点灯すればエンジン種別・確信度・
  新しいIN/STOP/TP1/2R水準を表示。非点灯なら理由(反転済み等)を表示。
- 利確/損切り位置の日次チェック: 当日のATR・直近5日高安値から
  「本日の構造的ストップ参考値」を毎回再計算(トレーリング判断用)。25日線・RSI・乖離率も表示。
- 含み損益は価格と%のみ(当初stop_priceの登録は不要)。
- 保有中に流動性がスクリーニング基準を下回った場合の警告も表示。

レポート最上部(結論ボックス直後)に「保有ポジション評価」ブロックとして出力されます。

## 実行(ローカル)

```bash
python -m pip install -r requirements.txt

# 合成データ+LINE非送信でパイプライン確認
SCREEN_DRYRUN=1 LINE_DRY_RUN=1 python main.py

# 本番相当(yfinance実データ、WORKER_URL未設定ならstdout出力)
python main.py
```

日本語フォント(PNG用): `sudo apt-get install fonts-noto-cjk`
(無い場合は `FONT_PATH` / `FONT_PATH_BOLD` で任意のCJKフォントを指定)

## 生成物(`out/`)

- `report_table_YYYY-MM-DD.png` — インフォグラフィック(LINE送信の主体)
- `report_YYYY-MM-DD.txt` — テキスト版(PNG失敗時のフォールバック)
- `plan_log_YYYY-MM-DD.csv` — 計画ログ(順流/逆流タグ・エンジン種別・セクター段階・
  非ファンダ性判定・レジームタグを含む・UTF-8 BOM)
- `result_log_template.csv` — 結果ログの空欄枠(非ファンダ群vsファンダ群の別を含む)
- `reject_ledger.csv` — 棄却台帳(追記式・N営業日後リターン自動追記)

## LINE配信(Worker + R2 方式・変更なし)

```
GitHub Actions(Python) → POST {WORKER_URL}/upload (multipart PNG)
  → Worker が R2 保存 → /img/<key> でHTTPS配信 → LINEへ image push
テキストのみ: POST {WORKER_URL}/ に {"text": "..."}
```

`src/worker.js` を既存Workerに上書きデプロイ。Secretsは従来命名を踏襲:
`LINE_TOKEN` / `LINE_USER_ID`(互換: `LINE_CHANNEL_ACCESS_TOKEN` / `LINE_TO`)、
任意 `UPLOAD_TOKEN`。R2 binding `REPORTS` は必須。

### 配信テスト手順

```bash
# 1. Worker設定確認
curl https://<worker>.workers.dev/health

# 2. テキスト到達
curl -X POST https://<worker>.workers.dev/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <UPLOAD_TOKEN(設定時のみ)>' \
  -d '{"text":"配信テスト"}'

# 3. 画像込み全経路(GitHub Actions の Run workflow で dryrun=1 でも可)
WORKER_URL=https://<worker>.workers.dev \
WORKER_AUTH_TOKEN=<UPLOAD_TOKEN(設定時のみ)> \
SCREEN_DRYRUN=1 python main.py
```

## GitHub Actions

`.github/workflows/screen.yml` — 平日 7:30 JST(cron `30 22 * * 0-4` UTC)。

- Secrets: `WORKER_URL`(必須)、`WORKER_AUTH_TOKEN`(Worker保護時)
- Variables: `ACCOUNT_EQUITY`(株数計算・任意)、`NIKKEI_VI_VALUE`(日経VI手動値・任意)
- 棄却台帳はリポジトリへ自動コミットで永続化(`permissions: contents: write`)

## 主要チューニング(環境変数)

| 変数 | 既定 | 意味 |
|---|---|---|
| `MIN_ADV_JPY` | 5億 | 流動性フィルタ(ADV・円) |
| `REL_Z_TH` / `REL_PCTL_TH` | 2.0 / 5 | エンジンA 業種内相対乖離の正規化主基準 |
| `DIP_MIN_DAYS` / `DIP_MAX_DAYS` | 2 / 5 | 押し目日数(1日反転はコスト負けで対象外) |
| `EVENT_OVERSHOOT_MULT` | 1.25 | イベント痕跡ありの深押し要求倍率 |
| `BUCKET_RELDEV_*` | 8/10/12 | 縮退時の相対乖離バケット閾値(%) |
| `REBOUND_MIN/MAX_DAYS` / `REBOUND_MIN_ATR` | 2/3 / 1.0 | エンジンS 逆流戻り売りのリバウンド条件 |
| `ENGINE_B_MONTHS` | 2,5,8,11 | PEAD(B疑い)を動かす決算月 |
| `FLOW_WINDOW` / `FLOW_WINDOW_LONG` | 5 / 10 | 資金循環マップの騰落率窓 |
| `REGIME_UP_SHARE` / `REGIME_DN_SHARE` | 0.72 / 0.28 | 全面高/全面安のレジーム判定閾値 |
| `STOP_ATR_MULT` (`_HIVOL`) | 1.5 (2.0) | 損切り幅ATR倍率 |
| `MAX_RISK_WIDTH_PCT` | 8 | リスク幅上限% |
| `HOLD_DAYS` / `CANDIDATE_EXPIRY_DAYS` / `TRAIL_DAYS` | 5 / 3 / 3 | 時間ストップ/未エントリー失効/トレール |
| `EST_COST_PCT` / `NET_R_FLOOR` | 0.15 / 1.6 | 概算コスト%とネットR床 |
| `MIN_EXEC_JPY` | 100万 | 最小実行サイズ(下回ると見送り) |
| `RISK_PCT_HIGH` / `RISK_PCT_MID` | 1.0 / 0.5 | 確信度→リスク%(固定フラクショナル) |
| `TOTAL_RISK_CAP` (`_HALF`) | 2.0 (1.0) | 同時総オープンリスク上限% |
| `VI_HALF_LOT` / `VI_WARN` / `VI_SEVERE` | 30 / 28 / 35 | VI閾値 |
| `NIKKEI_VI_VALUE` / `ACCOUNT_EQUITY` | — | 日経VI手動値 / 口座資金 |

## データについての注意

- 価格・指標は yfinance(Yahoo Finance系)の単一ソース。v5.0では単一ソースを許容し
  確信度は減点しませんが、本命は全件「仮点灯」で、確定はiSPEED照合+チャット側ゲート0/3後です。
- 資金循環マップの業種指数は構成銘柄の等ウェイト平均で代理(単一ソース参考)。
  段階判定・ブレッドスも参考値です。部門別売買動向(週次)はラグのため不使用。
- `positions.csv` の銘柄は `universe_jpx.csv` に存在する必要があります(名前・セクター解決のため)。

## 変更ファイル一覧(既存repoへ上書き)

- `main.py` / `mispricing/` 全体 / `src/worker.js` / `.github/workflows/screen.yml` /
  `requirements.txt` / `README.md`
- 上書きしない: `universe_jpx.csv` / `positions.csv` / `events.csv` / `wrangler.toml`
- 削除してよい: 旧 `utils/` `data/` `tests/`(参照なし)

---

## モメンタム・スクリーニング(全振り運用・2026年7月〜)

歪み系(上記)から方向転換し、**momentum/ 配下に完全独立**したモメンタム主体のスクリーニングを追加。
GitHub Actionsの実行対象は `main.py` から **`main_momentum.py`** に切り替え済み(=日々配信されるのは
モメンタム側のみ)。歪み系のコードは削除せず温存(将来の比較・再併用のための保険)。

### ハード制約(ユーザー明示指定・裁量による例外なし)

- 実保有は**最大3銘柄**
- **同一業種は1銘柄まで**(セクター分散必須)
- リスク%は確信度で可変にせず**固定0.5%**(3銘柄×0.5%=合計1.5%で総リスク上限2.0%内)
- **レジームフィルター(TOPIX)は例外なく厳守**:防御モードでは新規エントリー全停止

### 設計の骨格

1. **STEP1 レジームフィルター**: TOPIX(1306.T、フォールバック^N225)が200日線より上 かつ
   直近12ヶ月リターンがプラス → 攻撃モード。それ以外は防御モード(新規停止)。
2. **STEP2 候補プール**: 全ユニバースをモメンタム総合スコア(12-1モメンタム/52週高値近接度/
   対TOPIX相対強度/トレンド整列ボーナス)でランキングし、上位100銘柄(既定)をプール化。
3. **STEP3 3状態分類**:
   - **状態A(すでに流入)**: トレンド整列(終値>50日>150日>200日線)+ 10/20日線付近の押し目
   - **状態B(初動)**: VCP収縮(値幅圧縮)+ 出来高を伴うドンチアン・ブレイク
   - **状態C(流出)**: 50日線割れ → 新規対象外(保有ポジションの手仕舞い判定用、次回拡張予定)
4. **ポジション構成**: モメンタムスコア順に、3銘柄上限・1業種1銘柄までの制約で採用。
5. **出口設計**: 固定利確なし。シャンデリア・エグジット(直近22日高値−3×ATR)のトレーリングストップ
   のみ。初期ストップ(エントリー−2×ATR)は初日のみのR定義。

### ロングオンリー

モメンタムクラッシュ(急落後の反発局面で急激な損失が出るリスク)の大半は
「下げていた銘柄の急反発を売っている側」で起きるとされるため、ショートは実装せず
ロングオンリーとしている。状態Cは既存ポジションの手仕舞い判定にのみ使う設計。

### 実行(ローカル)

```bash
# 合成データ(レジーム強制トグル可)でパイプライン確認
MOM_DRYRUN_REGIME=on SCREEN_DRYRUN=1 LINE_DRY_RUN=1 ACCOUNT_EQUITY=10000000 python main_momentum.py
MOM_DRYRUN_REGIME=off SCREEN_DRYRUN=1 LINE_DRY_RUN=1 python main_momentum.py  # 防御モード確認用

# 本番相当
python main_momentum.py
```

### 生成物(`out_momentum/`)

- `momentum_report_table_YYYY-MM-DD.png` — インフォグラフィック
- `momentum_report_YYYY-MM-DD.txt` — テキスト版
- `momentum_plan_log_YYYY-MM-DD.csv` — 計画ログ
- `momentum_result_log_template.csv` / `momentum_reject_ledger.csv`

### 主要チューニング(環境変数・momentum/config.py)

| 変数 | 既定 | 意味 |
|---|---|---|
| `MOM_MAX_POSITIONS` | 3 | 実保有上限(★ハード制約) |
| `MOM_MAX_PER_SECTOR` | 1 | 同一業種の上限(★ハード制約) |
| `MOM_RISK_PCT_FIXED` | 0.5 | 固定リスク%(★確信度で可変にしない) |
| `MOM_POOL_SIZE` | 100 | 候補プールの規模(60〜120目安) |
| `REGIME_TICKER` / `_FALLBACK` | 1306.T / ^N225 | レジーム判定に使うTOPIX代理 |
| `ADX_TREND_TH` | 25 | トレンド強度の参考閾値(厳格なゲートではなくスコア/確認用) |
| `CHANDELIER_MULT` / `ATR_PERIOD` | 3.0 / 22 | シャンデリア・エグジットのATR倍率/期間 |
| `INITIAL_STOP_ATR_MULT` | 2.0 | 初期ストップのATR倍率 |
| `MIN_EXEC_JPY` | 100万円 | 最小実行サイズ(下回ると見送り) |

### 既知の注意点

- 状態C(流出)は現在「新規対象外」のみで、保有ポジションの自動手仕舞い判定への統合は未実装
  (position_check.py 相当の仕組みは歪み系のみ。次回拡張候補)。
- `MIN_EXEC_JPY`(100万円)と`MOM_RISK_PCT_FIXED`(0.5%)の組み合わせでは、口座資金や
  ATRの大きさによってサイズ過小で見送りになる候補が増える場合がある。ドライランでの
  検証では ACCOUNT_EQUITY 500万円時に全滅、1000万円時に採用が出た。実資金に応じて
  `MIN_EXEC_JPY` を下げるか、口座資金の設定を確認すること。
