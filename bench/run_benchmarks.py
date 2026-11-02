#!/usr/bin/env python3
"""Runs the full comparison and leaves machine-generated output in results/.

Every engine sees the same machine state because they run in one session: builds
alternate engine order between repetitions, and search interleaves engines inside one
process. Each step can be selected on its own, so a noisy step can be rerun alone.

Usage: run_benchmarks.py [--steps env,quality,kernels,build,search,scaling,keep,prefetch,
                                 prefetch_order,first_queries,pq] [--index-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "build" / "bench" / "bench" / "annvil_bench"
DATA = REPO / "data"
RESULTS = REPO / "results"

ENGINES = ["annvil", "annvil-bfs", "hnswlib", "hnswlib-tuned"]
EF_SWEEP = "10,20,40,80,120,160,240,320,480,640,800"
BUILD_THREADS = 8
# (dataset, M, ef_construction)
CONFIGS = [
    ("glove-25-angular", 16, 200),
    ("glove-25-angular", 32, 400),
    ("sift-128-euclidean", 16, 200),
]


def run(*args: object) -> None:
    command = [str(a) for a in args]
    print("+", " ".join(command[1:]), flush=True)
    subprocess.run(command, check=True, cwd=REPO)


def output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def reset(name: str) -> Path:
    path = RESULTS / name
    path.unlink(missing_ok=True)
    return path


def common(dataset: str, m: int, efc: int, index_dir: Path) -> list[object]:
    return [
        "--dataset", DATA / f"{dataset}.annb",
        "--M", m,
        "--efc", efc,
        "--index-dir", index_dir,
    ]  # fmt: skip


def step_env() -> None:
    harness = json.loads(output(str(BENCH), "info"))
    env = {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": output("sysctl", "-n", "machdep.cpu.brand_string"),
        "logical_cores": int(output("sysctl", "-n", "hw.ncpu")),
        "performance_cores": int(output("sysctl", "-n", "hw.perflevel0.physicalcpu")),
        "efficiency_cores": int(output("sysctl", "-n", "hw.perflevel1.physicalcpu")),
        "memory_gib": int(output("sysctl", "-n", "hw.memsize")) // 2**30,
        "cache_line_bytes": int(output("sysctl", "-n", "hw.cachelinesize")),
        "l1d_bytes_performance_core": int(output("sysctl", "-n", "hw.perflevel0.l1dcachesize")),
        "os": platform.platform(),
        "compiler": output("clang++", "--version").splitlines()[0],
        "hnswlib_version": "0.8.0",
        "build_threads": BUILD_THREADS,
        **harness,
    }
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def step_quality(_: Path) -> None:
    out = reset("parallel_quality.csv")
    out.write_text(
        "engine,threads,builds,recall_mean,recall_min,recall_max,"
        "builds_with_unreachable,worst_unreachable\n"
    )
    bench_dir = BENCH.parent
    run(bench_dir / "annvil_parallel_quality", "bottom-up", 100, out)
    run(bench_dir / "annvil_top_down_parallel_quality", "top-down", 100, out)


def step_kernels(_: Path) -> None:
    out = reset("kernels.csv")
    run(BENCH, "kernels", "--dims", "25,128,960", "--out", out)


def step_build(index_dir: Path) -> None:
    out = reset("build.csv")
    for repetition in range(2):
        order = ENGINES if repetition == 0 else list(reversed(ENGINES))
        for dataset, m, efc in CONFIGS:
            for engine in order:
                run(
                    BENCH, "build", *common(dataset, m, efc, index_dir),
                    "--engine", engine, "--threads", BUILD_THREADS, "--out", out,
                )  # fmt: skip


def step_search(index_dir: Path) -> None:
    out = reset("search.csv")
    for threads, passes in ((1, 5), (8, 5)):
        for dataset, m, efc in CONFIGS:
            run(
                BENCH, "search", *common(dataset, m, efc, index_dir),
                "--engines", ",".join(ENGINES), "--ef", EF_SWEEP,
                "--threads", threads, "--passes", passes, "--out", out,
            )  # fmt: skip


def step_scaling(index_dir: Path) -> None:
    out = reset("scaling.csv")
    for threads in (1, 2, 4, 6, 8, 10):
        for dataset, m, efc in (CONFIGS[0], CONFIGS[2]):
            run(
                BENCH, "search", *common(dataset, m, efc, index_dir),
                "--engines", "annvil,hnswlib-tuned,hnswlib", "--ef", "160",
                "--threads", threads, "--passes", 5, "--out", out,
            )  # fmt: skip


def step_keep(index_dir: Path) -> None:
    build = reset("keep_pruned_build.csv")
    search = reset("keep_pruned.csv")
    for dataset, m, efc in (CONFIGS[0], CONFIGS[2]):
        run(
            BENCH, "build", *common(dataset, m, efc, index_dir),
            "--engine", "annvil-keep", "--threads", BUILD_THREADS, "--out", build,
        )  # fmt: skip
        run(
            BENCH, "search", *common(dataset, m, efc, index_dir),
            "--engines", "annvil,annvil-keep", "--ef", EF_SWEEP,
            "--threads", 1, "--passes", 3, "--out", search,
        )  # fmt: skip


def step_prefetch(index_dir: Path) -> None:
    out = reset("prefetch.csv")
    for dataset, m, efc, lines in (
        ("sift-128-euclidean", 16, 200, "0,1,2,3,4"),
        ("glove-25-angular", 16, 200, "0,1"),
    ):
        run(
            BENCH, "prefetch", *common(dataset, m, efc, index_dir),
            "--lines", lines, "--ef", "40,160", "--out", out,
        )  # fmt: skip


def step_prefetch_order(index_dir: Path) -> None:
    batched = reset("prefetch_order_batched.csv")
    one_ahead = reset("prefetch_order_one_ahead.csv")
    variants = ((BENCH, batched), (BENCH.parent / "annvil_bench_one_ahead", one_ahead))
    # Lines per stored vector on this CPU: 128 bytes and 512 bytes in 128-byte lines.
    for (dataset, m, efc), lines in ((CONFIGS[0], 1), (CONFIGS[2], 4)):
        # Separate binaries cannot interleave inside one process, so alternate runs.
        for _ in range(5):
            for binary, out in variants:
                run(
                    binary, "prefetch", *common(dataset, m, efc, index_dir),
                    "--lines", lines, "--ef", "40,160", "--passes", 3, "--out", out,
                )  # fmt: skip


def step_first_queries(index_dir: Path) -> None:
    out = reset("first_queries.csv")
    for dataset, m, efc in (CONFIGS[0], CONFIGS[2]):
        for _ in range(5):
            for engine in ("annvil", "hnswlib"):
                run(
                    BENCH, "first-queries", *common(dataset, m, efc, index_dir),
                    "--engine", engine, "--out", out,
                )  # fmt: skip


def step_pq(_: Path) -> None:
    out = reset("pq.csv")
    sweeps = [
        ("glove-25-angular", 16, 200, "5,10,16"),
        ("sift-128-euclidean", 16, 200, "8,16,32"),
    ]
    for dataset, m, efc, pq_m in sweeps:
        run(
            BENCH, "pq", "--dataset", DATA / f"{dataset}.annb", "--M", m, "--efc", efc,
            "--pq-m", pq_m, "--ef", "64,128,256,512", "--rerank", "0,32,64,128,256",
            "--threads", BUILD_THREADS, "--out", out,
        )  # fmt: skip


STEPS = {
    "quality": step_quality,
    "kernels": step_kernels,
    "build": step_build,
    "search": step_search,
    "scaling": step_scaling,
    "keep": step_keep,
    "prefetch": step_prefetch,
    "prefetch_order": step_prefetch_order,
    "first_queries": step_first_queries,
    "pq": step_pq,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="env," + ",".join(STEPS))
    parser.add_argument("--index-dir", type=Path, default=REPO / "build" / "indexes")
    args = parser.parse_args()

    if not BENCH.exists():
        sys.exit("build the harness first: cmake --preset bench && cmake --build --preset bench")
    RESULTS.mkdir(exist_ok=True)
    for step in args.steps.split(","):
        print(f"== {step}", output("uptime"), flush=True)
        if step == "env":
            step_env()
        else:
            STEPS[step](args.index_dir)
    if args.index_dir == REPO / "build" / "indexes" and shutil.which("du"):
        print(output("du", "-sh", str(args.index_dir)))


if __name__ == "__main__":
    main()
