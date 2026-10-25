"""Local Character Creation persona explorer; run `python persona_explorer.py`.

Independently implemented exploration UI inspired by Lyra's neuron-explorer.
Inference is delegated to explicitly configured native Flare endpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

from persona_fleet import fetch
from persona_vectors import ModelSpec, VectorBank


class Explorer:
    def __init__(self, root: Path, config: dict):
        self.root, self.config = root, config
        self.model = ModelSpec.load(root / "data/persona_model.json")
        self.bank = VectorBank(root / "results", self.model)
        self.endpoints = config.get("endpoints", [])
        for endpoint in self.endpoints:
            if not isinstance(endpoint, dict) or not endpoint.get("url", "").startswith("http://127.0.0.1:"):
                raise ValueError("Endpoints must name explicitly forwarded localhost HTTP servers")
            if endpoint.get("model_identity") != self.model.identity or not endpoint.get("prepared_identity"):
                raise ValueError("Every endpoint requires the verified model and prepared-package identities")
        (root / "results").mkdir(exist_ok=True)
        self.database = root / "results/persona_explorer.sqlite"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, created REAL, state TEXT, input TEXT, output TEXT)")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.endpoint_lock = threading.Lock()
        self.endpoint_cursor = 0

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=15)

    def state(self) -> dict:
        metadata = json.loads((self.root / "data/persona_model.json").read_text())
        path = self.root / "results/persona_map.json"
        chart = json.loads(path.read_text()) if path.is_file() else {"neurons": [], "layers": [], "status": "Awaiting calibrated intervention results"}
        if chart.get("model_identity") not in (None, self.model.identity):
            raise ValueError("Map belongs to a different model")
        summary_path = self.root / "results/persona_baseline_summary.json"
        if not summary_path.exists():
            summary_path = self.root / "reports/persona_baseline.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        with self.connect() as db:
            comparisons = [{"id": row[0], "created": row[1], "state": row[2]}
                           for row in db.execute("SELECT id,created,state FROM comparisons ORDER BY created DESC LIMIT 20")]
        return {"model": metadata, "identity": self.model.identity, "map": chart,
                "baseline": summary, "comparisons": comparisons, "endpoints": len(self.endpoints),
                "calibrated_layers": [i for i in range(self.model.layers) if (self.root / "results" / f"layer_{i:02d}.npz").exists()]}

    def comparison(self, identity: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT state,input,output FROM comparisons WHERE id=?", (identity,)).fetchone()
        if row is None: raise ValueError("Comparison does not exist")
        return {"id": identity, "state": row[0], "input": json.loads(row[1]),
                "output": json.loads(row[2]) if row[2] else None}

    def submit(self, value: dict) -> str:
        if not self.endpoints: raise ValueError("No native Flare endpoints configured")
        if value.get("model_identity") != self.model.identity:
            raise ValueError("The selected recipe belongs to another model")
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
            raise ValueError("Enter a prompt of 1–20,000 characters")
        pushes = value.get("pushes", [])
        if not isinstance(pushes, list) or any(not isinstance(p, dict) for p in pushes):
            raise ValueError("Pushes must be a list of interventions")
        phase = value.get("phase", "all")
        # Validate all directions before publishing a queued job.
        self.bank.program(pushes, phase)
        seed = value.get("seed", 7035)
        if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError("Seed must be uint32")
        budget = value.get("max_tokens", 2048)
        if type(budget) is not int or not 256 <= budget <= 4096: raise ValueError("Token budget must be 256–4096")
        value = {"prompt": prompt, "pushes": pushes, "phase": phase, "seed": seed,
                 "max_tokens": budget, "thinking": bool(value.get("thinking", False)),
                 "model_identity": self.model.identity}
        identity = uuid.uuid4().hex
        with self.connect() as db:
            active = db.execute("SELECT COUNT(*) FROM comparisons WHERE state IN ('queued','running')").fetchone()[0]
            if active >= 4: raise ValueError("Four comparisons are already queued; wait for one to finish")
            db.execute("INSERT INTO comparisons VALUES (?,?,?,?,?)", (identity, time.time(), "queued", json.dumps(value), None))
        self.executor.submit(self.run_comparison, identity, value)
        return identity

    def run_comparison(self, identity: str, value: dict) -> None:
        with self.connect() as db: db.execute("UPDATE comparisons SET state='running' WHERE id=?", (identity,))
        def run(sign: int) -> dict:
            pushes = [{**p, "dose": float(p["dose"]) * sign} for p in value["pushes"]] if sign else []
            program, provenance = self.bank.program(pushes, value["phase"])
            with self.endpoint_lock:
                endpoint = self.endpoints[self.endpoint_cursor % len(self.endpoints)]; self.endpoint_cursor += 1
            descriptor = fetch(endpoint["url"] + "/v1/models", timeout=5)["data"][0]
            flare = descriptor.get("flare", {})
            if flare.get("prepared_identity") != endpoint["prepared_identity"]:
                raise ValueError("Native server prepared identity differs from the verified package")
            if program["sites"]:
                capability = flare.get("steering") or {}
                sites = {site["name"]: site["width"] for site in capability.get("sites", [])}
                if capability.get("protocol") != "additive-bf16-sites-v1":
                    raise ValueError("Native server does not advertise the required steering protocol")
                if len(program["sites"]) > capability.get("max_sites", 0):
                    raise ValueError("Recipe exceeds the native package's prepared site capacity")
                if any(sites.get(site["site"]) != len(site["vector"]) for site in program["sites"]):
                    raise ValueError("Recipe site or vector width is absent from the prepared package")
            model = descriptor["id"]
            request = {"model": model, "messages": [{"role": "user", "content": value["prompt"]}],
                       "seed": value["seed"], "temperature": .7, "top_p": .95, "top_k": 20,
                       "max_tokens": value["max_tokens"], "stream": False,
                       "chat_template_kwargs": {"enable_thinking": value["thinking"]}}
            if program["sites"]: request["steering"] = program
            response = fetch(endpoint["url"] + "/v1/chat/completions", request)
            choice = response["choices"][0]
            return {"sign": sign, "request": request, "endpoint": endpoint, "response": response, "provenance": provenance,
                    "complete": choice.get("finish_reason") == "stop" and bool(choice["message"].get("content"))}
        def protected_run(sign: int) -> dict:
            try:
                return run(sign)
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                return {"sign": sign, "error": str(error), "complete": False}
        try:
            with ThreadPoolExecutor(max_workers=3) as workers:
                output = list(workers.map(protected_run, [-1, 0, 1]))
            state = "failed" if any(row.get("error") for row in output) else "complete"
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
            output, state = {"error": str(error)}, "failed"
        with self.connect() as db:
            db.execute("UPDATE comparisons SET state=?,output=? WHERE id=?", (state, json.dumps(output), identity))


def handler(app: Explorer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, value: object, status: int = 200) -> None:
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path)
            try:
                if path.path == "/":
                    body = (app.root / "ui/persona_explorer.html").read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif path.path == "/atlas-logo.png":
                    body = (app.root / "ui/atlas-logo.png").read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif path.path == "/api/state": self.send(app.state())
                elif path.path == "/api/recipes":
                    self.send(json.loads((app.root / "data/lyra_recipes.json").read_text()))
                elif path.path == "/api/comparison":
                    self.send(app.comparison(parse_qs(path.query).get("id", [""])[0]))
                else: self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as error: self.send({"error": str(error)}, 400)

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send({"error": "Cross-origin requests are not accepted"}, 403); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100000: raise ValueError("Request size exceeds limit")
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict): raise ValueError("Expected JSON object")
                if self.path == "/api/compare": self.send({"id": app.submit(value)}, 202)
                else: self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as error: self.send({"error": str(error)}, 400)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    app = Explorer(args.root, config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    print(f"Character Creation: http://127.0.0.1:{args.port}", flush=True)
    try: server.serve_forever()
    finally: server.server_close(); app.executor.shutdown(wait=True)


if __name__ == "__main__": main()
