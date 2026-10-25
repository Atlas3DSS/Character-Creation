"""Launch six explicitly configured, independently leased capture or probe workers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def command(config: dict, shard: int, resume: bool, *, mode="capture", minutes=120, base_port=8930) -> list[str]:
    if mode not in {"capture", "probe"} or type(minutes) is not int or not 1 <= minutes <= 960:
        raise ValueError("Choose capture/probe and an explicit duration of 1..960 minutes")
    if mode == "probe" and (resume or not 1 <= base_port <= 65530):
        raise ValueError("Probe workers require six valid ports and no capture-resume option")
    root = config["remote_root"]
    sysroot = config["sysroot"]
    workers = config["compile_workers"][shard]
    environment = {
        "CUDA_VISIBLE_DEVICES": config["gpus"][shard], "CUDA_HOME": config["cuda_home"],
        "PATH": config["runtime_path"], "PKG_CONFIG_PATH": sysroot + "/usr/lib/x86_64-linux-gnu/pkgconfig",
        "PKG_CONFIG_SYSROOT_DIR": sysroot,
        "LD_LIBRARY_PATH": sysroot + "/usr/lib/x86_64-linux-gnu:" + config["cuda_home"] + "/lib64",
        "WB_HOLD_WAIT": "120", "OMP_NUM_THREADS": str(workers), "MKL_NUM_THREADS": str(workers),
        "TORCHINDUCTOR_COMPILE_THREADS": str(workers), "TORCHINDUCTOR_CACHE_DIR": config["inductor_cache"],
        "TRITON_CACHE_DIR": config["triton_cache"], "HF_HUB_OFFLINE": "1"}
    result = ["systemd-run", "--user", f"--unit=character-persona-{mode}-{shard}", "--collect",
              f"--property=RuntimeMaxSec={minutes * 60}", "--property=TimeoutStopSec=60",
              f"--working-directory={root}"]
    result.extend(f"--setenv={key}={value}" for key, value in environment.items())
    result.extend(["/usr/bin/python3", config["lease_wrapper"], "run", "character-creation",
        f"persona-{mode}-{shard}", "31", str(minutes), config["gpu_labels"][shard], "--",
        "taskset", "-c", config["cpu_sets"][shard], config["python"], root + f"/persona_{mode}.py", "run",
        "--config", root + "/persona_fleet.local.json",
        "--shard", str(shard), "--compile-workers", str(workers), "--flare", config["flare_source"],
        "--checkpoint", config["checkpoint"], "--heliopause", root + "/flare_heliopause.so",
        "--output", root + "/results", "--kernel-cache", config["kernel_cache"],
        "--cutlass-include", config["cutlass_include"]])
    if mode == "capture":
        result.extend(["--plan", root + "/results/persona_capture_plan.json"])
    else:
        result.extend(["--port", str(base_port + shard)])
    if resume:
        result.append("--resume")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("capture", "probe"), default="capture")
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--base-port", type=int, default=8930)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if any(len(config[key]) != 6 for key in ("gpus", "gpu_labels", "cpu_sets", "compile_workers")):
        raise ValueError("An explicit six-GPU configuration is required")
    if len(set(config["gpus"])) != 6:
        raise ValueError("Each shard requires a different physical GPU UUID")
    cpu_sets = []
    for value in config["cpu_sets"]:
        cpus = set()
        for section in value.split(","):
            ends = list(map(int, section.split("-")))
            cpus.update(range(ends[0], ends[-1]+1))
        if not cpus or any(cpus & other for other in cpu_sets):
            raise ValueError("Shard CPU affinities must be nonempty and disjoint")
        cpu_sets.append(cpus)
    ssh = ["ssh", "-o", "ControlMaster=no", "-o", "ControlPath=none", config["ssh_host"]]
    if not args.dry_run:
        actual = subprocess.check_output([*ssh, "hostname"], text=True).strip()
        if actual != config["hostname"]:
            raise ValueError("SSH target hostname differs from configured compute host")
    for shard in range(6):
        invocation = command(config, shard, args.resume, mode=args.mode, minutes=args.minutes, base_port=args.base_port)
        if args.dry_run:
            print(shlex.join([*ssh, shlex.join(invocation)]))
        else:
            subprocess.run([*ssh, shlex.join(invocation)], check=True)
            print(f"Launched {args.mode} shard {shard}, CPU affinity {config['cpu_sets'][shard]}", flush=True)


if __name__ == "__main__":
    main()
