export default {
  /**
   * Cloudflare Worker for forwarding text to LINE Push API.
   * Expects: POST / with JSON body: { "text": "..." }
   * Env vars (set in Cloudflare):
   *   - LINE_TOKEN   : Channel access token
   *   - LINE_USER_ID : Destination user ID
   */
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    let payload;
    try {
      payload = await request.json();
    } catch (e) {
      return new Response("Invalid JSON", { status: 400 });
    }

    const text = (payload && typeof payload.text === "string")
      ? payload.text
      : "";

    if (!text.trim()) {
      return new Response("Field 'text' is required", { status: 400 });
    }

    if (!env.LINE_TOKEN || !env.LINE_USER_ID) {
      return new Response("Missing LINE_TOKEN or LINE_USER_ID", { status: 500 });
    }

    const lineEndpoint = "https://api.line.me/v2/bot/message/push";

    const body = JSON.stringify({
      to: env.LINE_USER_ID,
      messages: [
        {
          type: "text",
          text,
        },
      ],
    });

    const res = await fetch(lineEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${env.LINE_TOKEN}`,
      },
      body,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return new Response(
        JSON.stringify({
          ok: false,
          status: res.status,
          statusText: res.statusText,
          error: errText,
        }),
        {
          status: 502,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    return new Response(
      JSON.stringify({ ok: true }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  },
};
