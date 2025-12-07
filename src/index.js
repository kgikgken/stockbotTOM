export default {
  async fetch(request, env) {
    let data = {};
    try {
      data = await request.json();
    } catch (e) {
      return new Response("Invalid JSON", { status: 400 });
    }

    if (!data || !data.text) {
      return new Response("No text", { status: 400 });
    }

    const LINE_TOKEN = env.LINE_TOKEN;
    const LINE_USER_ID = env.LINE_USER_ID;

    await fetch("https://api.line.me/v2/bot/message/push", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${LINE_TOKEN}`,
      },
      body: JSON.stringify({
        to: LINE_USER_ID,
        messages: [
          {
            type: "text",
            text: data.text,
          },
        ],
      }),
    });

    return new Response("OK");
  },
};