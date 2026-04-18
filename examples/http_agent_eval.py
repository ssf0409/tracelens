"""Evaluate an HTTP-backed agent in 10 minutes.

What this shows:
- `HTTPAPIAdapter` pointed at a real HTTP endpoint.
- A `JsonSchemaGrader` (GATE policy) enforcing output shape.
- 5 trials per task via `RunnerConfig(num_runs=5)`.
- Markdown + CI summary via `ReportGenerator`.

The "agent under test" here is a 20-line stdlib HTTPServer running in a
background thread. Swap its URL for your real service and the rest of
the script is unchanged — that's the point.

Run: `python examples/http_agent_eval.py`

Expected output (abridged):
    eval-kit: 2 tasks, 10 trials, pass_rate=100.0%, mean_score=1.0000, ...
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from eval_kit import (
    EvalSet,
    EvaluationRunner,
    HTTPAdapterConfig,
    HTTPAPIAdapter,
    JsonSchemaGrader,
    ReportGenerator,
    RunnerConfig,
    Task,
)


# --- The "agent": a stdlib HTTPServer that pretends to be your service. ---
class _EchoAgent(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        response = {"answer": body.get("a", 0) + body.get("b", 0), "ok": True}
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: object) -> None:  # Silence default stderr noise.
        return


def _run_server(srv: HTTPServer) -> None:
    srv.serve_forever()


async def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), _EchoAgent)
    thread = threading.Thread(target=_run_server, args=(server,), daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    # 1. Point HTTPAPIAdapter at the agent.
    adapter = HTTPAPIAdapter(
        HTTPAdapterConfig(base_url=base_url, endpoint="/", method="POST"),
    )

    # 2. Declare the expected output shape. GATE means any violation fails
    #    the trial outright, independent of other graders.
    grader = JsonSchemaGrader(
        "output_schema",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "integer"}, "ok": {"type": "boolean"}},
            "required": ["answer", "ok"],
        },
    )

    # 3. Two tasks, 5 runs each.
    eval_set = EvalSet(name="http-echo", tasks=[
        Task(name="add small", input_data={"a": 2, "b": 3}),
        Task(name="add large", input_data={"a": 1000, "b": 1234}),
    ])
    runner = EvaluationRunner(adapter, [grader], RunnerConfig(num_runs=5))

    batch = await runner.run(eval_set)

    gen = ReportGenerator()
    report = gen.build_report(batch)
    print(gen.render_ci_summary(report))
    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
