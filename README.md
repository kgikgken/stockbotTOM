# stockbotTOM

このZIPは、stockbotTOM の **動作する最小構成 + GitHub Actions 実行環境** をまとめたものです。

## 重要（あなたの既存repoへ上書きする場合）
- このZIP内の `events.csv / positions.csv / universe_jpx.csv / wrangler.toml / data/ / src/` は、
  **あなたのrepoに元から存在する版が正** です。
- あなたの運用データを上書きして消さないため、このZIPでは **雛形（プレースホルダ）** を同梱しています。
- **上書き対象は「変更ファイル一覧」に記載したものだけ** にしてください。

## 実行（ローカル）
```bash
python -m pip install -r requirements.txt
python main.py
```

## 生成物
- `out/report_table_YYYY-MM-DD.csv`
- `out/report_table_YYYY-MM-DD.svg`
- `out/report_table_YYYY-MM-DD.png`

## LINEに「画像」で送る（表PNG）

LINE Messaging API では画像を直接アップロードできず、**HTTPSで公開された画像URL**が必要です。
そのため、このリポジトリでは **Cloudflare Worker + R2** を使って下記を実現します。

1. GitHub Actions（Python）が `out/report_table_YYYY-MM-DD.png` を生成
2. そのPNGを Worker に multipart でアップロード
3. Worker が R2 に保存 → `/img/<key>` で配信（HTTPS）
4. Worker が LINE に **image message** を push

### Worker側の設定

- `wrangler.toml` の `[[r2_buckets]]` にある bucket 名（デフォルト `stockbot-reports`）を R2 で作成（または名前を変更）
- Worker の「Variables and Secrets」に以下を設定
  - ✅ 推奨（あなたの現状と同じ命名）
    - `LINE_TOKEN` : LINE公式アカウントのチャネルアクセストークン
    - `LINE_USER_ID` : 送信先（userId / groupId / roomId）
  - 互換（別名でもOK）
    - `LINE_CHANNEL_ACCESS_TOKEN` / `LINE_TO`
  - （任意）アップロード保護
    - `UPLOAD_TOKEN`（または `AUTH_TOKEN`） : 画像アップロード保護用トークン

#### 画像用ストレージ（必須）

表PNGを配信するために、Workerに **R2 bucket binding** が必要です。

- Cloudflareで R2 bucket を作成（例: `stockbot-reports`）
- Workerに binding を追加
  - 種別: R2
  - 変数名: `REPORTS`
  - Bucket: `stockbot-reports`

> 注意: これが未設定だと `/upload` が失敗し、LINEに画像は出ません（テキストだけ届く状態になります）。

### GitHub Actions（実行環境）側の設定

- `WORKER_URL` : デプロイした Worker のURL
- （任意）`WORKER_AUTH_TOKEN` : Workerに `UPLOAD_TOKEN` を設定した場合のみ
- （任意）`LINE_SEND_IMAGE=0` : 画像送信をOFF

## Workerコード

- Cloudflareの「Edit code」に貼り付けるなら `src/worker.js`（JS版）が楽です。
- Wranglerでデプロイするなら `wrangler.toml` + `src/worker.js` を使います。
