export class TradeLogDO {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    if (pathname === "/log" && request.method === "POST") {
      const body = await request.json();
      await this.state.storage.put(Date.now().toString(), body);
      return new Response("OK", { status: 200 });
    }

    if (pathname === "/logs" && request.method === "GET") {
      const logs = await this.state.storage.list();
      return new Response(JSON.stringify(logs), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response("Not Found", { status: 404 });
  }
}

export default {
  async fetch(request, env) {
    const id = env.TRADE_LOG.idFromName("main");
    const stub = env.TRADE_LOG.get(id);

    const url = new URL(request.url);
    if (url.pathname.startsWith("/log")) {
      return stub.fetch(request);
    }
    if (url.pathname.startsWith("/logs")) {
      return stub.fetch(request);
    }

    return new Response("OK", { status: 200 });
  }
};