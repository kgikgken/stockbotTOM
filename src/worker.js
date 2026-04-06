export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    if (request.method === "GET" && (path === "/health" || path.endsWith("/health"))) {
      return handleHealth(env);
    }
    if (request.method === "GET" && path.includes("/img/")) {
      return handleImageGet(url, env, path);
    }
    if (request.method === "POST" && path.endsWith("/upload")) {
      return handleUpload(request, env, url, path);
    }
    if (request.method === "POST" && (path === "/" || path.endsWith("/push"))) {
      if (!authOk(request, env)) {
        return json({ ok: false, error: "unauthorized" }, 401);
      }
      return handlePush(request, env);
    }
    return json({ ok: true, service: "stockbotTOM-worker" });
  },
};

function normalizePath(pathname) {
  const clean = String(pathname || "/").replace(/\/+/g, "/");
  if (clean === "/") return "/";
  return clean.replace(/\/+$/, "") || "/";
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function authOk(request, env) {
  const expected = env.PUSH_TOKEN || env.UPLOAD_TOKEN || env.AUTH_TOKEN || "";
  if (!expected) return true;
  const header = request.headers.get("authorization") || "";
  return header === `Bearer ${expected}`;
}

function stripTerminalPath(path, terminal) {
  const suffix = `/${terminal}`;
  if (path === suffix) return "";
  if (path.endsWith(suffix)) return path.slice(0, -suffix.length);
  return path;
}

function handleHealth(env) {
  return json({
    ok: true,
    service: "stockbotTOM-worker",
    reportsBinding: Boolean(env.REPORTS),
    lineTokenPresent: Boolean(env.LINE_TOKEN || env.LINE_CHANNEL_ACCESS_TOKEN),
    lineRecipientPresent: Boolean(env.LINE_USER_ID || env.LINE_TO),
    authEnabled: Boolean(env.PUSH_TOKEN || env.UPLOAD_TOKEN || env.AUTH_TOKEN),
  });
}

async function handleUpload(request, env, url, path) {
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
  const prefix = stripTerminalPath(path, "upload");
  return json({ ok: true, key, url: `${url.origin}${prefix}/img/${encodeURIComponent(key)}` });
}

async function handleImageGet(url, env, path) {
  if (!env.REPORTS) {
    return new Response("REPORTS binding missing", { status: 500 });
  }
  const marker = "/img/";
  const idx = path.lastIndexOf(marker);
  const key = idx >= 0 ? decodeURIComponent(path.slice(idx + marker.length)) : "";
  if (!key) {
    return new Response("not found", { status: 404 });
  }
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
