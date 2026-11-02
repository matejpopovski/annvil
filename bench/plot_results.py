#!/usr/bin/env python3
"""Turns results/*.csv into the charts and the tables quoted in the README.

Nothing is typed by hand: results/summary.md and results/plots/ are regenerated from
the CSVs, so the README can be checked against them line by line.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PLOTS = RESULTS / "plots"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": {"annvil": "#2a78d6", "hnswlib": "#eb6834", "hnswlib-tuned": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "series": {"annvil": "#3987e5", "hnswlib": "#d95926", "hnswlib-tuned": "#199e70"},
    },
}
MARKERS = {"annvil": "o", "hnswlib": "^", "hnswlib-tuned": "D"}
ENGINES = ["annvil", "annvil-bfs", "hnswlib-tuned", "hnswlib"]
# annvil-bfs is left off the chart: it lies on top of annvil, as summary.md shows.
PLOTTED = ["annvil", "hnswlib", "hnswlib-tuned"]
RECALL_TICKS = [0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
RECALL_FLOOR, RECALL_CEILING = 0.65, 0.9995
PANELS = [
    ("glove-25-angular", 16, "glove-25-angular, M=16"),
    ("glove-25-angular", 32, "glove-25-angular, M=32"),
    ("sift-128-euclidean", 16, "sift-128-euclidean, M=16"),
]
RECALL_TARGETS = [0.90, 0.95, 0.99]


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def curve(rows: list[dict], dataset: str, m: int, engine: str, threads: int, key="qps_best"):
    points = [
        (float(r["recall"]), float(r[key]), int(r["ef"]))
        for r in rows
        if r["dataset"] == dataset
        and int(r["M"]) == m
        and r["engine"] == engine
        and int(r["threads"]) == threads
    ]
    return sorted(points, key=lambda p: p[2])


def qps_at(points, target: float) -> float | None:
    """Interpolates log QPS between the two ef settings that bracket the target recall."""
    for (r0, q0, _), (r1, q1, _) in zip(points, points[1:], strict=False):
        if r0 <= target <= r1 and r1 > r0:
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) * (1 - t) + math.log(q1) * t)
    return None


def style(ax, theme) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)


def plot_recall_qps(rows, threads: int, theme_name: str) -> None:
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=theme["surface"])
    for ax, (dataset, m, title) in zip(axes, PANELS, strict=True):
        style(ax, theme)
        for engine in PLOTTED:
            # A logit axis cannot place recall 1.0, and past three nines the curves
            # only measure how close the rounding is to it.
            points = [
                p
                for p in curve(rows, dataset, m, engine, threads)
                if RECALL_FLOOR <= p[0] <= RECALL_CEILING
            ]
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=theme["series"][engine],
                linewidth=2,
                marker=MARKERS[engine],
                markersize=6,
                markeredgecolor=theme["surface"],
                markeredgewidth=1,
                label=engine,
            )
            # Direct labels go on the panel where the curves are furthest apart.
            if ax is axes[2]:
                ax.annotate(
                    engine,
                    (points[-1][0], points[-1][1]),
                    xytext=(4, -15 if engine == "hnswlib" else 9),
                    textcoords="offset points",
                    ha="right",
                    color=theme["text"],
                    fontsize=9,
                )
        ax.set_xscale("logit")
        ax.set_xlim(RECALL_FLOOR, RECALL_CEILING)
        ax.set_xticks(RECALL_TICKS, [f"{t:g}" for t in RECALL_TICKS])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(title, color=theme["text"], fontsize=11, loc="left")
        ax.set_xlabel("recall@10 (logit scale)", color=theme["muted"], fontsize=10)
    label = f"queries per second, {threads} thread(s)"
    axes[0].set_ylabel(label, color=theme["muted"], fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=theme["text"],
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    suffix = "" if theme_name == "light" else "-dark"
    fig.savefig(PLOTS / f"recall-qps-{threads}t{suffix}.png", dpi=150, facecolor=theme["surface"])
    plt.close(fig)


def plot_prefetch(rows, theme_name: str) -> None:
    theme = THEMES[theme_name]
    selected = [r for r in rows if r["dataset"].startswith("sift") and r["ef"] == "40"]
    if not selected:
        return
    selected.sort(key=lambda r: int(r["lines"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor=theme["surface"])
    style(ax, theme)
    ax.grid(False, axis="x")
    labels = [r["lines"] for r in selected]
    values = [float(r["qps_best"]) for r in selected]
    bars = ax.bar(labels, values, width=0.55, color=theme["series"]["annvil"])
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:,.0f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=theme["text"],
            fontsize=9,
        )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("sift-128, ef=40, 1 thread", color=theme["text"], fontsize=11, loc="left")
    ax.set_xlabel("cache lines prefetched per vector (a vector spans 4)", color=theme["muted"])
    ax.set_ylabel("queries per second", color=theme["muted"])
    fig.tight_layout()
    suffix = "" if theme_name == "light" else "-dark"
    fig.savefig(PLOTS / f"prefetch{suffix}.png", dpi=150, facecolor=theme["surface"])
    plt.close(fig)


def fmt(value: float | None) -> str:
    return "not reached" if value is None else f"{value:,.0f}"


def median_ms(rows: list[dict], key: str) -> float:
    return statistics.median(float(r[key]) * 1000 for r in rows)


def mib(value: float) -> str:
    return f"{value / 2**20:,.0f}"


def summary() -> str:
    lines = ["# Benchmark summary", ""]
    lines += ["Generated by bench/plot_results.py from results/*.csv.", ""]
    env = RESULTS / "environment.json"
    if env.exists():
        e = json.loads(env.read_text())
        lines += [
            f"{e['cpu']}, {e['memory_gib']} GiB, {e['os']}, {e['compiler']}.",
            f"annvil: `{e['annvil_flags']}`. hnswlib {e['hnswlib_version']}: "
            f"`{e['hnswlib_flags']}`. Kernel: {e['kernel']}.",
            "",
        ]

    search = read("search.csv")
    for threads in (1, 8):
        lines += [f"## QPS at fixed recall, {threads} thread(s)", ""]
        lines += ["| dataset | M | recall | annvil | annvil-bfs | hnswlib-tuned | hnswlib |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for dataset, m, _ in PANELS:
            for target in RECALL_TARGETS:
                values = [
                    fmt(qps_at(curve(search, dataset, m, e, threads), target)) for e in ENGINES
                ]
                lines += [f"| {dataset} | {m} | {target:.2f} | " + " | ".join(values) + " |"]
        lines.append("")

    lines += ["## Graph quality at equal ef (1 thread)", ""]
    lines += ["| dataset | M | ef | engine | recall | strict recall | distances/query |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for r in search:
        if r["threads"] == "1" and r["ef"] in ("40", "160"):
            lines += [
                f"| {r['dataset']} | {r['M']} | {r['ef']} | {r['engine']} | "
                f"{float(r['recall']):.4f} | {float(r['recall_ids']):.4f} | "
                f"{float(r['distance_computations']):,.0f} |"
            ]
    lines.append("")

    grouped = defaultdict(list)
    for r in read("build.csv"):
        grouped[(r["dataset"], r["M"], r["engine"])].append(r)
    if grouped:
        lines += ["## Build and memory (8 threads, best of repetitions)", ""]
        lines += [
            "| dataset | M | engine | build s | peak RSS MiB | engine MiB | file MiB | "
            "load during |"
        ]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for (dataset, m, engine), rows in sorted(grouped.items()):
            best = min(rows, key=lambda r: float(r["build_seconds"]))
            lines += [
                f"| {dataset} | {m} | {engine} | {float(best['build_seconds']):.1f} | "
                f"{mib(float(best['peak_rss_bytes']))} | {mib(float(best['engine_bytes']))} | "
                f"{mib(float(best['file_bytes']))} | "
                f"{float(best['load_before']):.1f} to {float(best['load_after']):.1f} |"
            ]
        lines.append("")

    seen = {}
    for r in search:
        seen[(r["dataset"], r["M"], r["engine"])] = float(r["load_seconds"])
    if seen:
        lines += ["## Time to load a saved index", ""]
        lines += ["| dataset | M | engine | seconds |", "|---|---|---|---|"]
        for (dataset, m, engine), seconds in sorted(seen.items()):
            lines += [f"| {dataset} | {m} | {engine} | {seconds:.3f} |"]
        lines.append("")

    quality = read("parallel_quality.csv")
    if quality:
        lines += ["## Parallel build quality, clustered synthetic data, 4,000 points", ""]
        lines += [
            "| engine | threads | builds | mean recall | min | max | builds with unreachable "
            "nodes | worst unreachable |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in quality:
            lines += [
                f"| {r['engine']} | {r['threads']} | {r['builds']} | "
                f"{float(r['recall_mean']):.4f} | {float(r['recall_min']):.3f} | "
                f"{float(r['recall_max']):.3f} | {r['builds_with_unreachable'] or 'n/a'} | "
                f"{r['worst_unreachable'] or 'n/a'} |"
            ]
        lines.append("")

    kernels = read("kernels.csv")
    if kernels:
        lines += ["## Distance kernels on L1-resident data (ns per call)", ""]
        lines += ["| dim | metric | annvil SIMD | hnswlib (auto-vectorized) | scalar |"]
        lines += ["|---|---|---|---|---|"]
        table = defaultdict(dict)
        for r in kernels:
            table[(int(r["dim"]), r["metric"])][r["kernel"]] = float(r["ns_per_call"])
        for (dim, metric), row in sorted(table.items()):
            lines += [
                f"| {dim} | {metric} | {row['annvil-simd']:.2f} | {row['hnswlib']:.2f} | "
                f"{row['scalar']:.2f} |"
            ]
        lines.append("")

    prefetch = read("prefetch.csv")
    if prefetch:
        lines += ["## QPS against prefetch depth (1 thread)", ""]
        lines += ["| dataset | ef | lines | best QPS | median QPS |", "|---|---|---|---|---|"]
        for r in sorted(prefetch, key=lambda r: (r["dataset"], int(r["ef"]), int(r["lines"]))):
            lines += [
                f"| {r['dataset']} | {r['ef']} | {r['lines']} | "
                f"{float(r['qps_best']):,.0f} | {float(r['qps_median']):,.0f} |"
            ]
        lines.append("")

    batched, one_ahead = read("prefetch_order_batched.csv"), read("prefetch_order_one_ahead.csv")
    if batched and one_ahead:
        lines += ["## Prefetch order, full-vector prefetch in both (1 thread, best QPS)", ""]
        lines += ["| dataset | ef | all neighbours first | one ahead (hnswlib's loop) | ratio |"]
        lines += ["|---|---|---|---|---|"]
        best = defaultdict(float)
        for name, rows in (("batched", batched), ("ahead", one_ahead)):
            for r in rows:
                key = (r["dataset"], int(r["ef"]), name)
                best[key] = max(best[key], float(r["qps_best"]))
        for dataset, ef in sorted({(r["dataset"], int(r["ef"])) for r in batched}):
            a, b = best[(dataset, ef, "batched")], best[(dataset, ef, "ahead")]
            lines += [f"| {dataset} | {ef} | {a:,.0f} | {b:,.0f} | {a / b:.2f} |"]
        lines.append("")

    first = read("first_queries.csv")
    if first:
        lines += ["## Fresh process: open, then 1,000 queries at ef=100 (median of 5, ms)", ""]
        lines += [
            "| dataset | engine | open | first 1,000 | second 1,000 | open + first 1,000 |",
            "|---|---|---|---|---|---|",
        ]
        grouped_first = defaultdict(list)
        for r in first:
            grouped_first[(r["dataset"], r["engine"])].append(r)
        for (dataset, engine), rows in sorted(grouped_first.items()):
            load_ms = median_ms(rows, "load_seconds")
            first_ms = median_ms(rows, "first_queries_seconds")
            second_ms = median_ms(rows, "second_queries_seconds")
            lines += [
                f"| {dataset} | {engine} | {load_ms:,.1f} | {first_ms:,.1f} | "
                f"{second_ms:,.1f} | {load_ms + first_ms:,.1f} |"
            ]
        lines.append("")

    scaling = read("scaling.csv")
    if scaling:
        lines += ["## Read scaling at ef=160 (best QPS)", ""]
        lines += ["| dataset | threads | annvil | hnswlib-tuned | hnswlib |"]
        lines += ["|---|---|---|---|---|"]
        table = defaultdict(dict)
        for r in scaling:
            table[(r["dataset"], int(r["threads"]))][r["engine"]] = float(r["qps_best"])
        for (dataset, threads), row in sorted(table.items()):
            lines += [
                f"| {dataset} | {threads} | {row['annvil']:,.0f} | "
                f"{row['hnswlib-tuned']:,.0f} | {row['hnswlib']:,.0f} |"
            ]
        lines.append("")

    keep = read("keep_pruned.csv")
    if keep:
        lines += ["## keepPrunedConnections (annvil-keep) against the default (1 thread)", ""]
        lines += ["| dataset | M | recall | annvil | annvil-keep |", "|---|---|---|---|---|"]
        for dataset, m in (("glove-25-angular", 16), ("sift-128-euclidean", 16)):
            for target in RECALL_TARGETS:
                values = [fmt(qps_at(curve(keep, dataset, m, e, 1), target)) for e in ENGINES[:1]]
                values.append(fmt(qps_at(curve(keep, dataset, m, "annvil-keep", 1), target)))
                lines += [f"| {dataset} | {m} | {target:.2f} | " + " | ".join(values) + " |"]
        lines.append("")

    pq = read("pq.csv")
    if pq:
        lines += ["## Product quantization, best QPS reaching each recall", ""]
        lines += [
            "| dataset | pq m | recall >= | ef | rerank | recall | QPS | " "RAM for traversal MiB |"
        ]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for dataset in sorted({r["dataset"] for r in pq}):
            for pq_m in sorted({int(r["pq_m"]) for r in pq if r["dataset"] == dataset}):
                rows = [r for r in pq if r["dataset"] == dataset and int(r["pq_m"]) == pq_m]
                for target in RECALL_TARGETS:
                    hits = [r for r in rows if float(r["recall"]) >= target]
                    if not hits:
                        lines += [f"| {dataset} | {pq_m} | {target:.2f} | not reached | | | | |"]
                        continue
                    best = max(hits, key=lambda r: float(r["qps_best"]))
                    # Exact traversal needs vectors in RAM; code-only traversal does not.
                    ram = float(best["link_bytes"]) + (
                        float(best["vector_bytes"]) if pq_m == 0 else float(best["code_bytes"])
                    )
                    lines += [
                        f"| {dataset} | {pq_m} | {target:.2f} | {best['ef']} | {best['rerank']} | "
                        f"{float(best['recall']):.4f} | {float(best['qps_best']):,.0f} | "
                        f"{mib(ram)} |"
                    ]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    search = read("search.csv")
    prefetch = read("prefetch.csv")
    for theme in THEMES:
        for threads in (1, 8):
            plot_recall_qps(search, threads, theme)
        plot_prefetch(prefetch, theme)
    (RESULTS / "summary.md").write_text(summary())


if __name__ == "__main__":
    main()
