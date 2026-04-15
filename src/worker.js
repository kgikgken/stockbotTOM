// Cloudflare Worker (JavaScript version)
//
// Purpose:
// - Accept text-only notifications (POST JSON: {text})
// - Accept image uploads (POST multipart/form-data: image + optional text + optional key)
// - Host uploaded images via GET /img/<key>
// - Push to LINE Messaging API as:
//     - text message(s)
//     - image message (URL)
//
// Env vars (Cloudflare Worker "Variables and Secrets"):
// - LINE_TOKEN (or LINE_CHANNEL_ACCESS_TOKEN)
// - LINE_USER_ID (or LINE_TO)
// - (optional) UPLOAD_TOKEN (or AUTH_TOKEN)
// - (required for images) REPORTS: R2 bucket binding
// - (optional) PUBLIC_BASE_URL: base URL for image links

function json(data, init = {}) {
  const headers = new Headers(init.headers || {});
  headers.set("content-type", "application/json; charset=utf-8");
  return new Response(JSON.stringify(data, null, 2), { ...init, headers });
}

function ok(data = { ok: true }) {
  return json(data, { status: 200 });
}

function badRequest(message, extra = {}) {
  return json({ ok: false, error: message, ...extra }, { status: 400 });
}

function unauthorized() {
  return json({ ok: false, error: "unauthorized" }, { status: 401 });
}

function notFound() {
  return json({ ok: false, error: "not found" }, { status: 404 });
}

function needConfig() {
  return json(
    {
      ok: false,
      error:
        "missing LINE token/to or REPORTS binding. Set (LINE_CHANNEL_ACCESS_TOKEN & LINE_TO) or (LINE_TOKEN & LINE_USER_ID).",
    },
    { status: 500 },
  );
}

function getLineToken(env) {
  return ((env.LINE_CHANNEL_ACCESS_TOKEN || env.LINE_TOKEN) || "").trim();
}

function getLineTo(env) {
  return ((env.LINE_TO || env.LINE_USER_ID) || "").trim();
}

function requireAuth(req, env) {
  const token = ((env.UPLOAD_TOKEN || env.AUTH_TOKEN) || "").trim();
  if (!token) return true;

  const auth = (req.headers.get("authorization") || "").trim();
  const xAuth = (req.headers.get("x-auth-token") || "").trim();
  return auth === `Bearer ${token}` || xAuth === token;
}

async function pushToLine(env, messages) {
  const token = getLineToken(env);
  const to = getLineTo(env);
  if (!token || !to) {
    return needConfig();
  }

  const payload = { to, messages };

  const res = await fetch("https://api.line.me/v2/bot/message/push", {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const text = await res.text();
  if (!res.ok) {
    return json(
      { ok: false, error: "LINE push failed", status: res.status, detail: text },
      { status: 502 },
    );
  }

  return ok({ ok: true, line: text ? safeJson(text) : "" });
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function handleNotify(request, env) {
  // POST JSON: { text: "..." }
  let body;
  try {
    body = await request.json();
  } catch {
    return badRequest("invalid json");
  }
  const text = (body?.text || "").toString().trim();
  if (!text) return badRequest("missing text");
  return await pushToLine(env, [{ type: "text", text }]);
}

async function handleUpload(request, env) {
  // POST multipart/form-data:
  // - image: file (required)
  // - text: string (optional)
  // - key: string (optional)
  if (!requireAuth(request, env)) return unauthorized();
  if (!env.REPORTS) return needConfig();

  const form = await request.formData();
  const file = form.get("image");
  if (!file || typeof file === "string") {
    return badRequest("missing image file field 'image'");
  }

  const text = (form.get("text") || "").toString().trim();
  const rawKey = (form.get("key") || "").toString().trim();
  const key = rawKey ? `reports/${rawKey}` : `reports/report_${Date.now()}.png`;

  const contentType = file.type || "image/png";
  await env.REPORTS.put(key, file.stream(), {
    httpMetadata: { contentType },
  });

  const baseUrl = (env.PUBLIC_BASE_URL || new URL(request.url).origin).replace(/\/$/, "");
  const imgUrl = `${baseUrl}/img/${encodeURIComponent(key)}`;

  const messages = [
    { type: "image", originalContentUrl: imgUrl, previewImageUrl: imgUrl },
  ];
  if (text) messages.push({ type: "text", text });

  const pushed = await pushToLine(env, messages);
  // If LINE push failed, return that error.
  if (pushed.status !== 200) return pushed;

  return ok({ ok: true, key, url: imgUrl });
}

async function handleImg(request, env) {
  if (!env.REPORTS) return needConfig();

  const u = new URL(request.url);
  const keyEnc = u.pathname.replace(/^\/img\//, "");
  if (!keyEnc) return notFound();
  const key = decodeURIComponent(keyEnc);

  const obj = await env.REPORTS.get(key);
  if (!obj) return notFound();

  const headers = new Headers();
  headers.set("content-type", obj.httpMetadata?.contentType || "application/octet-stream");
  headers.set("cache-control", "public, max-age=604800");
  return new Response(obj.body, { headers });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    if (request.method === "GET") {
      if (pathname === "/" || pathname === "/health") return ok();
      if (pathname.startsWith("/img/")) return await handleImg(request, env);
      return notFound();
    }

    if (request.method === "POST") {
      const ct = request.headers.get("content-type") || "";
      if (ct.startsWith("multipart/form-data")) {
        return await handleUpload(request, env);
      }
      return await handleNotify(request, env);
    }

    return notFound();
  },
};