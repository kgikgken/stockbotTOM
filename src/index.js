export default {
  async fetch(request, env) {
    try {
      const { text } = await request.json();

      const res = await fetch(
        "https://api.line.me/v2/bot/message/push",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${env.LINE_TOKEN}`
          },
          body: JSON.stringify({
            to: env.LINE_USER_ID,
            messages: [{ type: "text", text }]
          })
        }
      );

      return new Response("OK", { status: 200 });
    } catch (e) {
      return new Response("ERROR:" + e.message, { status: 500 });
    }
  }
}