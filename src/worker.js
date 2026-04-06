export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname.startsWith("/img/")) {
      return handleImageGet(url, env);
    }
    if (request.method === "POST" && url.pathname === "/upload") {
      return handleUpload(request, env, url);
    }
    if (request.method === "POST" && (url.pathname === "/push" || url.pathname === "/")) {
      return handlePush(request, env);
    }
    return json({ ok: true, service: "stockbotTOM-worker" });
  },
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function authOk(request, env) {
  const expected = env.UPLOAD_TOKEN || env.AUTH_TOKEN || "";
  if (!expected) return true;
  const header = request.headers.get("authorization") || "";
  return header === `Bearer ${expected}`;
}

async function handleUpload(request, env, url) {
  if (!authOk(request, env)) {
    return json({ ok: false, error: "unauthorized" }, 401);
  }
  if (!env.REPORTS) {
    return json({ ok: false, error: "REPORTS binding missing" }, 500);
  }
  const form = await request.formData();
  const file = form.get("file");
  if (!file || typeof file === "string") {
    return json({ ok: false, error: "file missing" }, 400);
  }
  const safeName = String(form.get("path") || file.name || `report-${Date.now()}.png`).replace(/[^a-zA-Z0-9._/-]/g, "_");
  const key = `${new Date().toISOString().slice(0, 10)}/${safeName}`;
  await env.REPORTS.put(key, await file.arrayBuffer(), {
    httpMetadata: { contentType: file.type || "image/png" },
  });
  return json({ ok: true, key, url: `${url.origin}/img/${encodeURIComponent(key)}` });
}

async function handleImageGet(url, env) {
  if (!env.REPORTS) {
    return new Response("REPORTS binding missing", { status: 500 });
  }
  const key = decodeURIComponent(url.pathname.replace(/^\/img\//, ""));
  const obj = await env.REPORTS.get(key);
  if (!obj) {
    return new Response("not found", { status: 404 });
  }
  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("cache-control", "public, max-age=300");
  return new Response(obj.body, { headers });
}

async function linePush(env, messages) {
  const token = env.LINE_TOKEN || env.LINE_CHANNEL_ACCESS_TOKEN;
  const to = env.LINE_USER_ID || env.LINE_TO;
  if (!token || !to) {
    return { ok: true, dryrun: true, reason: "LINE credentials missing", messages };
  }
  const res = await fetch("https://api.line.me/v2/bot/message/push", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ to, messages }),
  });
  const text = await res.text();
  return { ok: res.ok, status: res.status, body: text.slice(0, 500) };
}

async function handlePush(request, env) {
  const payload = await request.json().catch(() => ({}));
  const text = String(payload.text || "").trim();
  const imageUrls = Array.isArray(payload.imageUrls) ? payload.imageUrls.filter(Boolean) : [];
  const messages = [];
  if (text) {
    messages.push({ type: "text", text: text.slice(0, 5000) });
  }
  for (const url of imageUrls.slice(0, 5)) {
    messages.push({
      type: "image",
      originalContentUrl: url,
      previewImageUrl: url,
    });
  }
  if (!messages.length) {
    return json({ ok: true, skipped: true, reason: "no_content" });
  }
  const result = await linePush(env, messages);
  return json(result, result.ok ? 200 : 500);
}
