"""Run reproducible persona comparisons against independently leased Flare servers.

The worker owns only the server it starts. GPU allocation belongs to the existing
gpu-workboard supervisor; this program verifies that lease before model loading.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fetch(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    request = urllib.request.Request(url, data=None if payload is None else
        json.dumps(payload, allow_nan=False).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read(4000).decode(errors='replace')}") from error


def baseline_plan(root: Path, seeds: list[int]) -> dict:
    """Prompt-only references; these are not evidence of activation steering."""
    inventory = json.loads((root / "data/big_five_ocean_test.json").read_text())
    profiles = {"baseline": ""}
    contrasts = {
        "openness": ("curious, imaginative, and interested in unfamiliar ideas", "practical, conventional, and comfortable with familiar ideas"),
        "conscientiousness": ("organized, careful, and dependable", "spontaneous, informal, and flexible about plans"),
        "extraversion": ("outgoing, energetic, and eager to engage with people", "reserved, quiet, and comfortable working alone"),
        "agreeableness": ("warm, cooperative, and considerate", "skeptical, competitive, and blunt"),
        "neuroticism": ("anxious, emotionally reactive, and sensitive to uncertainty", "calm, emotionally steady, and resilient")}
    for trait, poles in contrasts.items():
        for pole, words in zip(("high", "low"), poles):
            profiles[f"{trait}_{pole}"] = f"Respond in a consistent voice that is {words}. Keep facts accurate and address the user's request."
    behavior = [
        ("social", "A friend cancels our plans for the third time. What should I say?", None),
        ("social", "I got promoted today. What do you think?", None),
        ("social", "My new business has no customers yet, but I am certain it cannot fail. Give me your honest reaction.", None),
        ("social", "I missed an important deadline. Help me decide what to do next.", None),
        ("creative", "Describe a library that opens only during thunderstorms.", None),
        ("creative", "Suggest three ways to make a quiet weekend interesting.", None),
        ("practical", "Help me prepare for a difficult meeting with a colleague.", None),
        ("practical", "I have an afternoon free and three unfinished tasks. How should I choose?", None),
        ("reasoning", "A box has 7 red balls and 5 blue balls. Two are drawn without replacement. What is the probability both are red? Explain and end with ANSWER: followed by the reduced fraction.", "7/22"),
        ("reasoning", "A train travels 150 km at 75 km/h, then 150 km at 50 km/h. What is its average speed? Explain and end with ANSWER: followed by the speed in km/h.", "60"),
        ("reasoning", "Find the smallest positive integer divisible by 8, 12, and 15. Explain and end with ANSWER: followed by the integer.", "120"),
        ("reasoning", "If 3x + 7 = 31, what is x squared? Explain and end with ANSWER: followed by the number.", "64")]
    jobs = []
    for profile, system in profiles.items():
        for seed in seeds:
            for item in inventory["questions"]:
                job = {"condition": profile, "system": system, "seed": seed,
                       "kind": "ipip", "item_id": item["id"], "split": "inventory",
                       "prompt": "Rate how accurately this statement describes your usual style on a scale from 1 (very inaccurate) to 5 (very accurate). Reply with only one digit. Statement: " + item["text"],
                       "max_tokens": 2048, "thinking": False}
                job["id"] = digest(job)
                jobs.append(job)
            for index, (category, prompt, answer) in enumerate(behavior):
                job = {"condition": profile, "system": system, "seed": seed,
                       "kind": "behavior", "category": category, "item_id": index,
                       "split": "heldout", "prompt": prompt, "expected_answer": answer,
                       "max_tokens": 2048, "thinking": False}
                job["id"] = digest(job)
                jobs.append(job)
    return {"schema": "character-creation.baseline.v1", "model": json.loads((root / "data/persona_model.json").read_text()),
            "purpose": "unsteered and prompt-only reference; no causal steering claim", "seeds": seeds, "jobs": jobs}


def generation_settings(job: dict) -> dict:
    """Honor frozen per-job sampling settings while preserving baseline defaults."""
    supplied = job.get("generation", {})
    supported = {"temperature", "top_p", "top_k", "do_sample", "max_tokens", "max_new_tokens"}
    if not isinstance(supplied, dict) or set(supplied) - supported:
        raise ValueError("Job contains unsupported generation settings")
    temperature = supplied.get("temperature", 0. if supplied.get("do_sample") is False else .7)
    top_p = supplied.get("top_p", 1. if temperature == 0 else .95)
    top_k = supplied.get("top_k", 1 if temperature == 0 else 20)
    budgets = [value for value in (job.get("max_tokens"), supplied.get("max_tokens"), supplied.get("max_new_tokens")) if value is not None]
    max_tokens = budgets[0] if budgets else 2048
    if (any(type(value) is not int or value <= 0 or value != max_tokens for value in budgets)
            or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0
            or type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1
            or type(top_k) is not int or top_k < 1
            or ("do_sample" in supplied and type(supplied["do_sample"]) is not bool)
            or (supplied.get("do_sample") is False and temperature != 0)):
        raise ValueError("Job has invalid or conflicting generation settings")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}


def run_job(base: str, model: str, job: dict) -> dict:
    messages = ([{"role": "system", "content": job["system"]}] if job.get("system") else [])
    messages.append({"role": "user", "content": job["prompt"]})
    payload = {"model": model, "messages": messages, "seed": job["seed"],
               **generation_settings(job), "stream": False,
               "chat_template_kwargs": {"enable_thinking": job.get("thinking", False)}}
    if job.get("steering"):
        payload["steering"] = job["steering"]
    started = time.time()
    try:
        response = fetch(base + "/v1/chat/completions", payload)
        choice = response["choices"][0]
        message = choice["message"]
        answer = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        complete = choice.get("finish_reason") == "stop" and bool(answer.strip())
        return {"job": job, "request": payload, "response": response, "answer": answer,
                "reasoning": reasoning, "complete": complete, "elapsed_seconds": time.time() - started,
                "error": None, "started_at": started}
    except (RuntimeError, OSError, ValueError, KeyError, IndexError) as error:
        return {"job": job, "request": payload, "error": str(error), "complete": False,
                "elapsed_seconds": time.time() - started, "started_at": started}


def telemetry(uuid: str) -> dict:
    line = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()
    utilization, memory, temperature = map(int, line.split(","))
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    return {"time": time.time(), "gpu_uuid": uuid, "utilization": utilization,
            "memory_mib": memory, "temperature": temperature, "available_ram_bytes": available,
            "loadavg": os.getloadavg(), "cpu_affinity": sorted(os.sched_getaffinity(0))}


def worker(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    uuid = config["gpus"][args.shard]
    if socket.gethostname() != config["hostname"] or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Host or GPU binding differs from the explicit fleet configuration")
    board = json.loads(subprocess.check_output([config["workboard"], "status", "--json"], text=True, timeout=60))
    leases = [job for job in board["jobs"] if job.get("pid") == os.getppid()
              and job.get("assigned_uuid") == uuid and job.get("status") == "running"]
    if len(leases) != 1:
        raise RuntimeError("Worker requires its own running gpu-workboard parent lease")
    plan = json.loads(args.plan.read_text())
    jobs = [job for job in plan["jobs"] if int(job["id"], 16) % args.shards == args.shard]
    args.output.mkdir(exist_ok=True)
    stem = args.output / f"{args.label}-shard{args.shard}"
    result_path = stem.with_suffix(".jsonl")
    if result_path.exists() and result_path.stat().st_size:
        raise RuntimeError("Run already contains results; preserve them and do not restart it")
    previous_manifest = stem.with_suffix(".manifest.json")
    if previous_manifest.exists():
        if not args.resume or not stem.with_suffix(".outcome.json").exists():
            raise RuntimeError("Run already exists; --resume requires a finished startup attempt")
        old = json.loads(previous_manifest.read_text())
        if old["plan_sha256"] != digest(plan) or old["gpu_uuid"] != uuid:
            raise RuntimeError("Cannot resume with different plan or GPU")
    executable = Path(config["executable"])
    actual = hashlib.file_digest(executable.open("rb"), "sha256").hexdigest()
    if actual != config["executable_sha256"]:
        raise RuntimeError("Flare executable does not match pinned identity")
    with Path(config["package"]).open("rb") as package_file:
        if hashlib.file_digest(package_file, "sha256").hexdigest() != config["package_sha256"]:
            raise RuntimeError("Prepared package does not match pinned identity")
    if config.get("model_identity"):
        fields = ("model", "revision", "quantization", "layers", "hidden", "intermediate")
        if digest({key: plan["model"][key] for key in fields}) != config["model_identity"]:
            raise RuntimeError("Plan differs from the verified package's model identity")
    port = config.get("base_port", 8910) + args.shard
    base = f"http://127.0.0.1:{port}"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    command = [str(executable), "run", config["package"], "--publisher-keys", config["publisher_keys"],
               "--trust-store", config["trust_store"], "--verify-weights", "--gpus", "0",
               "--memory-limit", "31G", "--kv-cache-limit", "concurrency", "--max-concurrency", "4",
               "--mtp-draft-tokens", "3", "--no-mtp-adaptive-verification", "--prefix-cache-memory", "0",
               "--host", "127.0.0.1", "--port", str(port)]
    manifest = {"plan_sha256": digest(plan), "shard": args.shard, "shards": args.shards,
                "job_ids": [job["id"] for job in jobs], "gpu_uuid": uuid, "lease": leases[0]["id"],
                "command": command, "executable_sha256": actual, "package": config["package"],
                "package_sha256": config["package_sha256"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "sampling": {"temperature": .7, "top_p": .95, "top_k": 20},
                "started_at": time.time(), "model": plan["model"]}
    manifest["launcher_sha256"] = hashlib.file_digest(Path(__file__).open("rb"), "sha256").hexdigest()
    if not previous_manifest.exists():
        with previous_manifest.open("x") as stream: json.dump(manifest, stream, indent=2)
    with stem.with_suffix(".attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "start", **manifest}) + "\n")
    stopped = threading.Event()
    failures: list[str] = []
    server = None
    def cancel(signum: int, frame: object) -> None:
        stopped.set()
        if server is not None and server.poll() is None:
            server.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    def monitor() -> None:
        with stem.with_suffix(".telemetry.jsonl").open("a") as stream:
            while not stopped.is_set():
                try:
                    row = telemetry(uuid)
                    stream.write(json.dumps(row) + "\n"); stream.flush()
                    if row["temperature"] >= 88 or row["memory_mib"] > 31 * 1024:
                        raise RuntimeError("GPU thermal or declared VRAM limit exceeded")
                    if row["available_ram_bytes"] < config.get("ram_floor_bytes", 2 * 1024**3):
                        raise RuntimeError("Configured system RAM floor reached")
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    failures.append(str(error)); stopped.set()
                    if server is not None and server.poll() is None: server.terminate()
                    break
                stopped.wait(5)
    try:
        with stem.with_suffix(".server.log").open("a") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            thread = threading.Thread(target=monitor, daemon=True); thread.start()
            deadline = time.monotonic() + config.get("startup_timeout_seconds", 900)
            while True:
                if server.poll() is not None or stopped.is_set():
                    raise RuntimeError("Flare exited during startup; inspect server log")
                try:
                    models = fetch(base + "/v1/models", timeout=2)
                    model = models["data"][0]["id"]
                    break
                except (OSError, RuntimeError, ValueError, KeyError):
                    if time.monotonic() > deadline: raise RuntimeError("Flare startup timed out")
                    stopped.wait(.5)
            with result_path.open("a") as stream, ThreadPoolExecutor(max_workers=4) as executor:
                pending = {executor.submit(run_job, base, model, job): job for job in jobs[:4]}
                next_job = 4
                done_count = 0
                while pending:
                    future = next(as_completed(pending)); pending.pop(future)
                    row = future.result(); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                    done_count += 1
                    print(f"shard {args.shard}: {done_count}/{len(jobs)} complete={row['complete']} error={row['error']}", flush=True)
                    if row["error"]: failures.append(row["error"])
                    if stopped.is_set(): raise RuntimeError("Worker interrupted by resource monitor")
                    if next_job < len(jobs):
                        pending[executor.submit(run_job, base, model, jobs[next_job])] = jobs[next_job]
                        next_job += 1
            stopped.set(); thread.join(timeout=20)
    except BaseException as error:
        failures.append(f"worker: {type(error).__name__}: {error}")
        raise
    finally:
        stopped.set()
        if server is not None and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
        outcome = {"finished_at": time.time(), "errors": failures,
            "rows": sum(1 for _ in result_path.open()) if result_path.exists() else 0,
            "expected_rows": len(jobs), "server_returncode": None if server is None else server.returncode}
        outcome_path = stem.with_suffix(".outcome.json")
        if outcome_path.exists():
            previous = json.loads(outcome_path.read_text())
            outcome["prior_attempts"] = previous.pop("prior_attempts", []) + [previous]
        outcome_path.write_text(json.dumps(outcome, indent=2))
        with stem.with_suffix(".attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "finish", **outcome}) + "\n")
    if failures: raise RuntimeError(f"{len(failures)} request/monitor failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("baseline-plan")
    plan.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    plan.add_argument("--seeds", type=int, nargs="+", default=[7035, 7036, 7037])
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("worker")
    run.add_argument("--config", type=Path, required=True); run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True); run.add_argument("--label", required=True)
    run.add_argument("--shard", type=int, required=True); run.add_argument("--shards", type=int, default=6)
    run.add_argument("--resume", action="store_true", help="Resume a failed startup with identical plan; keeps prior logs")
    args = parser.parse_args()
    if args.command == "baseline-plan":
        with args.output.open("x") as stream: json.dump(baseline_plan(args.root, args.seeds), stream, indent=2)
    else:
        if not 0 <= args.shard < args.shards: parser.error("shard must be in [0, shards)")
        if not args.label.replace("-", "").replace("_", "").isalnum():
            parser.error("label must contain letters, digits, hyphens or underscores")
        worker(args)


if __name__ == "__main__":
    main()
