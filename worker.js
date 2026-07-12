/**
 * stockbot-worker — R2画像配信 + LINE Messaging API push
 *
 * Endpoints:
 *   GET  /health        設定状態の確認(トークン値は返さない)
 *   POST /upload        multipart {file, caption?} → R2保存 → LINEへ image push
 *   POST /  or /push    JSON {"text": "..."}       → LINEへ text push
 *   GET  /img/<key>     R2から画像配信(HTTPS公開URL)
 *
 * Secrets/Vars:
 *   LINE_TOKEN   (互換: LINE_CHANNEL_ACCESS_TOKEN)
 *   LINE_USER_ID (互換: LINE_TO)
 *   UPLOAD_TOKEN (互換: AUTH_TOKEN) — 任意。設定時は Authorization: Bearer 必須
 * Bindings:
 *   REPORTS — R2 bucket
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    try {
      if (request.method === "GET" && path === "/health") {
        return json({
          ok: true,
          line_token: !!(env.LINE_TOKEN || env.LINE_CHANNEL_ACCESS_TOKEN),
          line_to: !!(env.LINE_USER_ID || env.LINE_TO),
          r2: !!env.REPORTS,
          auth_required: !!(env.UPLOAD_TOKEN || env.AUTH_TOKEN),
        });
      }

      if (request.method === "GET" && path.startsWith("/img/")) {
        if (!env.REPORTS) return json({ ok: false, error: "R2 binding REPORTS 未設定" }, 500);
        const key = decodeURIComponent(path.slice(5));
        const obj = await env.REPORTS.get(key);
        if (!obj) return new Response("not found", { status: 404 });
        return new Response(obj.body, {
          headers: {
            "Content-Type": obj.httpMetadata?.contentType || "image/png",
            "Cache-Control": "public, max-age=604800",
          },
        });
      }

      if (request.method === "POST" && path === "/upload") {
        const authErr = checkAuth(request, env);
        if (authErr) return authErr;
        if (!env.REPORTS) return json({ ok: false, error: "R2 binding REPORTS 未設定" }, 500);

        const form = await request.formData();
        const file = form.get("file");
        if (!file || typeof file === "string") {
          return json({ ok: false, error: "multipart field 'file' が必要" }, 400);
        }
        const caption = form.get("caption") || "";
        const safeName = (file.name || "report.png").replace(/[^\w.\-]/g, "_");
        const key = `${Date.now()}_${safeName}`;
        await env.REPORTS.put(key, file.stream(), {
          httpMetadata: { contentType: file.type || "image/png" },
        });
        const imgUrl = `${url.origin}/img/${encodeURIComponent(key)}`;

        const messages = [];
        if (caption) messages.push({ type: "text", text: String(caption).slice(0, 4900) });
        messages.push({
          type: "image",
          originalContentUrl: imgUrl,
          previewImageUrl: imgUrl,
        });
        const line = await pushLine(env, messages);
        return json({ ok: line.ok, url: imgUrl, key, line }, line.ok ? 200 : 502);
      }

      if (request.method === "POST" && (path === "/" || path === "/push")) {
        const authErr = checkAuth(request, env);
        if (authErr) return authErr;
        const body = await request.json().catch(() => ({}));
        const text = String(body.text || "").slice(0, 4900);
        if (!text) return json({ ok: false, error: "text が空" }, 400);
        const line = await pushLine(env, [{ type: "text", text }]);
        return json({ ok: line.ok, line }, line.ok ? 200 : 502);
      }

      return json({ ok: false, error: "not found" }, 404);
    } catch (e) {
      return json({ ok: false, error: String(e) }, 500);
    }
  },
};

function checkAuth(request, env) {
  const required = env.UPLOAD_TOKEN || env.AUTH_TOKEN;
  if (!required) return null;
  const h = request.headers.get("Authorization") || "";
  const tok = h.startsWith("Bearer ") ? h.slice(7) : new URL(request.url).searchParams.get("token");
  if (tok === required) return null;
  return json({ ok: false, error: "unauthorized" }, 401);
}

async function pushLine(env, messages) {
  const token = env.LINE_TOKEN || env.LINE_CHANNEL_ACCESS_TOKEN;
  const to = env.LINE_USER_ID || env.LINE_TO;
  if (!token || !to) {
    return { ok: false, error: "LINE_TOKEN / LINE_USER_ID 未設定" };
  }
  const r = await fetch("https://api.line.me/v2/bot/message/push", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ to, messages }),
  });
  const body = await r.text();
  return { ok: r.ok, status: r.status, body: body.slice(0, 300) };
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
