# Benchmark notes

What was measured, on what, and how far to trust it. The generated tables are in
[`results/summary.md`](results/summary.md). The raw CSVs and `environment.json` sit beside it.

## Machine

Apple M3 Max, 10 performance and 4 efficiency cores, 36 GB, 128-byte cache lines, 128 KB L1D on
performance cores, macOS 26.6, Apple clang 21.0.0. annvil is built with `-O3 -mcpu=native`, and
hnswlib 0.8.0 with `-O3 -ffast-math -mcpu=native`. Nothing was pinned to cores, since macOS offers
no thread affinity.

## The machine was shared

Other parallel jobs were using this CPU during the benchmarks, sometimes heavily, and that affects
the numbers. This is how it was handled.

The first full run went ahead when load looked low, and load then rose. It is kept as a record in
[`results/run-under-load/`](results/run-under-load) and is not the source of any figure in the
README. The final run waited for the 1-minute load average to drop below 10 before each step.

| search run | mean load, per configuration | best / worst QPS across passes, median | worst |
|---|---|---|---|
| under load (kept for reference) | 17.5 to 73.0 | 1.40 | 5.14 |
| final | 6.9 to 14.5 | 1.16 | 1.75 |

So "quiet" here means quieter, not idle. Load of 7 to 14 on 14 cores includes the benchmark's own
threads, but other work was still running, and absolute QPS is lower than an idle machine would
give. As one example, `prefetch_order` ran later at load near 5 and shows 10,160 QPS for the same
sift-128 ef=160 setting that `prefetch` measured at 8,188. Compare numbers within a table, and do
not treat absolute QPS as a property of the hardware.

Ratios between engines are what the method protects:

- Search interleaves all engines inside one process, per ef and per pass, rotating the order, so a
  load spike lands on a whole pass and not on one engine. Each cell records its mean load.
- At 1 thread the ratios barely moved between the two runs, mostly within 5%. For example,
  annvil / hnswlib at recall 0.95 on sift-128 was 2.99 in the final run and 2.86 under load.
- At 8 threads, heavy load had inflated annvil's lead (2.76 on glove M=32 under load, 2.10 final).
  Contention hurt hnswlib more. The final run is the one reported, and the 8-thread results are
  the least stable numbers here.

Builds cannot be interleaved, so each ran twice, in opposite engine order, and the best is
reported. Load during each build is in `results/build.csv`, and it varied: annvil's best sift-128
build ran at load near 20 and `hnswlib-tuned`'s at load near 12, which does not favour annvil. A
sibling job pushed load to 65 at the tail of the second repetition, which is one reason best-of-two
is used. Treat build ratios as approximate. hnswlib's stock and tuned sift builds took 51 s and
35 s, a gap that is at least partly noise.

## Method

- **Datasets.** glove-25-angular and sift-128-euclidean from ann-benchmarks. Query vectors and, for
  angular data, base vectors are normalized once by the harness. annvil normalizes queries again
  inside its timed path, a small cost hnswlib does not pay.
- **Parameters.** Same M, efConstruction and ef sweep (10 to 800) for every engine. Build uses 8
  threads through one shared `parallel_for`, so schedulers do not differ.
- **Recall.** ann-benchmarks' definition (`plotting/metrics.py` upstream): a result counts if its
  true distance is within 1e-3 of the k-th true neighbour's. The harness recomputes distances from
  returned labels, so no engine's own distances are trusted. Strict id overlap is in the CSV too.
- **QPS.** Best of 5 passes over all 10,000 queries, with the median and worst alongside in
  `search.csv`. QPS at a target recall is log-linear interpolation between adjacent ef values.
- **Latency.** A separate pass, so per-query timers stay out of the QPS measurement. For example,
  at ef=160 on sift-128, one thread, p50 / p99 is 121 / 214 us for annvil and 347 / 904 us for
  hnswlib.
- **Memory.** Peak RSS comes from a separate build process and includes the harness's own copy of
  the vectors and the mapped dataset, so use it to compare engines, not as an absolute. "Engine
  MiB" is exact for annvil. For hnswlib it is computed from its allocations, so it is an estimate.
- **Warm cache.** All runs had the indexes in the page cache. `first_queries` is warm too, because
  evicting the cache needs root. Cold reads from disk were not measured for any engine.

## Attribution

Each experiment changes one thing and keeps the rest fixed.

| question | how it was measured | result |
|---|---|---|
| Are the graphs equivalent? | recall and distance counts at equal ef | yes, within 0.004 and 4.5% |
| Do kernels matter? | L1-resident microbenchmark, `kernels.csv` | barely; hnswlib ties or wins at some sizes |
| Does prefetch depth matter? | same index, 0 to 4 lines, `prefetch.csv` | 2.2x from 1 line to 4 |
| Does prefetch timing matter? | annvil with hnswlib's loop, `bench/one_ahead_prefetch.patch` | 1.46x to 1.84x |
| Does BFS reordering help? | `annvil-bfs` | no, 0.94x to 1.02x |
| Does keepPruned help? | `annvil-keep` | slightly, 1% to 9% |
| Is linking order safe? | 100 builds, `bench/top_down_linking.patch` | top-down orphans nodes |

The largest distance-count gap, 4.5%, is at ef=10 on glove M=32, where annvil computes fewer. It
is a constant 30 computations per query at every ef in that configuration, so it falls to 0.2% by
ef=800. On sift-128 the gap is under 1% at every ef.

The worst unreachable count for top-down linking varies between studies. It is 61 in the recorded
run and reached 254 in an earlier exploratory run, so the recorded figure is not an upper bound.

## Reproducing

```sh
uv run python bench/run_benchmarks.py                       # everything
uv run python bench/run_benchmarks.py --steps search,build  # selected steps
uv run python bench/plot_results.py                         # regenerate summary and charts
```

Each step replaces its own CSV. Steps: `env`, `quality`, `kernels`, `build`, `search`, `scaling`,
`keep`, `prefetch`, `prefetch_order`, `first_queries`, `pq`. The steps `search`, `scaling`,
`prefetch`, `prefetch_order` and `first_queries` need the saved indexes from `build`.

The recorded steps did not all use one set of indexes. `quality` and `kernels` come from the first
run, at load 6 to 14, and were not repeated. The final `search`, `scaling`, `prefetch` and `keep`
used indexes built during the first run. `build` was then rerun, at the end because it takes
longest, with the same parameters, and `prefetch_order` and `first_queries` used those newer
indexes. The graphs differ only by parallel build nondeterminism.
