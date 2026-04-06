export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = normalizePath(url.pathname);
    const contentType = request.headers.get("content-type") || "";

    if (request.method === "GET" && (path === "/health" || path.endsWith("/health") || path === "/")) {
      return handleHealth(env);
    }
    if (request.method === "GET" && path.includes("/img/")) {
      return handleImageGet(url, env, path);
    }

    if (request.method === "POST" && contentType.startsWith("multipart/form-data")) {
      if (path === "/" || path.endsWith("/upload") || isLegacyBasePath(path)) {
        return handleUpload(request, env, url, path);
      }
    }

    if (request.method === "POST") {
      if (path === "/" || path.endsWith("/push") || isLegacyBasePath(path)) {
        return handlePush(request, env, path);
      }
    }

    return json({ ok: false, error: "not_found", service: "stockbotTOM-worker" }, 404);
  },
};

function normalizePath(pathname) {
  const clean = String(pathname || "/").replace(/\/+/g, "/");
  if (clean === "/") return "/";
  return clean.replace(/\/+$/, "") || "/";
}

function isLegacyBasePath(path) {
  return path !== "/push" && path !== "/upload" && !path.endsWith("/push") && !path.endsWith("/upload") && !path.includes("/img/");
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function authToken(env) {
  return String(env.WORKER_AUTH_TOKEN || env.PUSH_TOKEN || env.UPLOAD_TOKEN || env.AUTH_TOKEN || "").trim();
}

function authOk(request, env) {
  const expected = authToken(env);
  if (!expected) return true;
  const header = (request.headers.get("authorization") || "").trim();
  const xAuth = (request.headers.get("x-auth-token") || "").trim();
  return header === `Bearer ${expected}` || xAuth === expected;
}

function lineToken(env) {
  return String(
    env.LINE_TOKEN || env.LINE_CHANNEL_ACCESS_TOKEN || env.LINE_ACCESS_TOKEN || env.CHANNEL_ACCESS_TOKEN || "",
  ).trim();
}

function lineTo(env) {
  return String(
    env.LINE_USER_ID || env.LINE_TO || env.LINE_TARGET_ID || env.TARGET_ID || env.USER_ID || env.TO || "",
  ).trim();
}

function stripTerminalPath(path, terminal) {
  const suffix = `/${terminal}`;
  if (path === suffix) return "";
  if (path.endsWith(suffix)) return path.slice(0, -suffix.length);
  return path;
}

function buildImageUrl(url, path, key) {
  const prefix = stripTerminalPath(path, "upload").replace(/\/+$/, "");
  return `${url.origin}${prefix}/img/${encodeURIComponent(key)}`;
}

function handleHealth(env) {
  return json({
    ok: true,
    service: "stockbotTOM-worker",
    reportsBinding: Boolean(env.REPORTS),
    lineTokenPresent: Boolean(lineToken(env)),
    lineRecipientPresent: Boolean(lineTo(env)),
    authEnabled: Boolean(authToken(env)),
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
  const file = form.get("file") || form.get("image");
  if (!file || typeof file === "string") {
    return json({ ok: false, error: "file missing" }, 400);
  }

  const rawName = String(form.get("path") || form.get("key") || file.name || `report-${Date.now()}.png`);
  const safeName = rawName.replace(/[^a-zA-Z0-9._/-]/g, "_");
  const key = safeName.includes("/") ? safeName : `${new Date().toISOString().slice(0, 10)}/${safeName}`;

  await env.REPORTS.put(key, await file.arrayBuffer(), {
    httpMetadata: { contentType: file.type || "image/png" },
  });

  const imgUrl = buildImageUrl(url, path, key);
  const text = String(form.get("text") || "").trim();

  const legacyMode = path === "/" || isLegacyBasePath(path);
  if (legacyMode) {
    const messages = [
      {
        type: "image",
        originalContentUrl: imgUrl,
        previewImageUrl: imgUrl,
      },
    ];
    if (text) {
      messages.push({ type: "text", text: text.slice(0, 5000) });
    }
    const pushed = await linePush(env, messages);
    return json(pushed, pushed.ok ? 200 : 500);
  }

  return json({ ok: true, key, url: imgUrl }, 200);
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
  const token = lineToken(env);
  const to = lineTo(env);
  if (!token || !to) {
    return { ok: false, status: 500, body: "LINE credentials missing" };
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

async function handlePush(request, env, path) {
  const requireAuth = path.endsWith("/push");
  if (requireAuth && !authOk(request, env)) {
    return json({ ok: false, error: "unauthorized" }, 401);
  }

  const payload = await request.json().catch(() => ({}));
  const text = String(payload.text || "").trim();
  const imageUrls = Array.isArray(payload.imageUrls) ? payload.imageUrls.filter(Boolean) : [];
  const messages = [];
  if (text) {
    messages.push({ type: "text", text: text.slice(0, 5000) });
  }
  for (const imageUrl of imageUrls.slice(0, 5)) {
    const clean = String(imageUrl || "").trim();
    if (!clean) continue;
    messages.push({
      type: "image",
      originalContentUrl: clean,
      previewImageUrl: clean,
    });
  }
  if (!messages.length) {
    return json({ ok: true, skipped: true, reason: "no_content" });
  }
  const result = await linePush(env, messages);
  return json(result, result.ok ? 200 : 500);
}
