// Benchmark harness for annvil and hnswlib on ann-benchmarks datasets.
//
//   build   Builds one engine in its own process, so peak RSS means something.
//   search  Loads all engines into one process and interleaves them per ef and per
//           pass. A load spike from elsewhere on the machine then lands on a whole
//           pass rather than on one engine, which is what keeps the comparison fair
//           on a shared box.
//   pq      Sweeps product-quantization settings on annvil.
//   prefetch  Measures QPS against prefetch depth on one saved annvil index.
//   kernels   Times distance functions on cache-resident data.
//   first-queries  Measures open plus the first queries in a fresh process.
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "annvil/dataset.hpp"
#include "annvil/index.hpp"
#include "annvil/kernels.hpp"
#include "annvil/parallel.hpp"
#include "engine.hpp"

namespace annvil::bench {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double load_average() {
  double load[1] = {0.0};
  return getloadavg(load, 1) == 1 ? load[0] : -1.0;
}

std::size_t peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);  // Bytes on macOS, KiB on Linux.
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
}

class AnnvilEngine final : public Engine {
 public:
  AnnvilEngine(const EngineConfig& config, bool reorder, bool keep_pruned)
      : config_(config), reorder_(reorder), keep_pruned_(keep_pruned) {}

  void build(const float* vectors, std::size_t n, std::size_t threads) override {
    IndexOptions options;
    options.dim = config_.dim;
    options.metric = config_.metric;
    options.capacity = config_.capacity;
    options.M = config_.M;
    options.ef_construction = config_.ef_construction;
    options.keep_pruned = keep_pruned_;
    index_ = std::make_unique<Index>(options);
    index_->add_batch({vectors, n * config_.dim}, threads);
    if (reorder_) {
      index_->reorder();
    }
  }
  void save(const std::filesystem::path& path) override { index_->save(path); }
  void load(const std::filesystem::path& path) override {
    index_ = std::make_unique<Index>(Index::open(path));
  }
  void set_ef(std::size_t ef) override { ef_ = ef; }

  std::size_t search(const float* query, std::size_t k, std::uint64_t* labels) const override {
    thread_local std::vector<Neighbor> out;
    out.resize(k);
    const std::size_t found = index_->search({query, config_.dim}, ef_, out);
    for (std::size_t i = 0; i < found; ++i) {
      labels[i] = out[i].label;
    }
    return found;
  }

  std::size_t search_counted(const float* query, std::size_t k, std::uint64_t* labels,
                             std::uint64_t* distance_computations) const override {
    thread_local std::vector<Neighbor> out;
    out.resize(k);
    SearchStats stats;
    const std::size_t found = index_->search({query, config_.dim}, ef_, out, &stats);
    for (std::size_t i = 0; i < found; ++i) {
      labels[i] = out[i].label;
    }
    *distance_computations += stats.distance_computations;
    return found;
  }

  std::size_t memory_bytes() const override { return index_->memory().total(); }

 private:
  EngineConfig config_;
  bool reorder_;
  bool keep_pruned_;
  std::size_t ef_ = 10;
  std::unique_ptr<Index> index_;
};

std::unique_ptr<Engine> make_engine(const std::string& name, const EngineConfig& config) {
  if (name == "annvil") {
    return std::make_unique<AnnvilEngine>(config, false, false);
  }
  if (name == "annvil-bfs") {
    return std::make_unique<AnnvilEngine>(config, true, false);
  }
  if (name == "annvil-keep") {
    return std::make_unique<AnnvilEngine>(config, false, true);
  }
  if (name == "hnswlib") {
    return make_hnswlib_engine(config);
  }
  if (name == "hnswlib-tuned") {
    return make_hnswlib_tuned_engine(config);
  }
  throw std::invalid_argument("unknown engine: " + name);
}

std::vector<std::string> split(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  for (std::string part; std::getline(stream, part, ',');) {
    parts.push_back(part);
  }
  return parts;
}

std::vector<std::size_t> split_sizes(const std::string& text) {
  std::vector<std::size_t> values;
  for (const std::string& part : split(text)) {
    values.push_back(std::stoull(part));
  }
  return values;
}

// ann-benchmarks' "recall": a result counts if its true distance is within 1e-3 of the
// k-th true neighbour's. Distances are recomputed here from the returned labels, so
// no engine's own distance values are trusted. See plotting/metrics.py upstream.
struct RecallScorer {
  const Dataset& dataset;
  std::vector<float> rows;  // Train vectors, normalized for the angular metric.
  std::vector<float> queries;
  std::size_t k;

  float distance(const float* a, const float* b) const {
    if (dataset.metric() == Metric::L2) {
      return std::sqrt(kernels::l2_sq_scalar(a, b, dataset.dim()));
    }
    return 1.0f - kernels::dot_scalar(a, b, dataset.dim());
  }

  // Returns {epsilon recall, strict id-overlap recall}.
  std::pair<double, double> score(const std::vector<std::uint64_t>& labels,
                                  const std::vector<std::uint32_t>& found) const {
    const std::size_t dim = dataset.dim();
    std::size_t within = 0;
    std::size_t overlap = 0;
    for (std::size_t q = 0; q < dataset.test_count(); ++q) {
      const float threshold = dataset.gt_distances(q)[k - 1] + 1e-3f;
      const std::int32_t* truth = dataset.gt_neighbors(q);
      for (std::size_t i = 0; i < found[q]; ++i) {
        const std::uint64_t label = labels[q * k + i];
        if (distance(&queries[q * dim], &rows[label * dim]) <= threshold) {
          ++within;
        }
        overlap += std::find(truth, truth + k, static_cast<std::int32_t>(label)) != truth + k;
      }
    }
    const double total = static_cast<double>(dataset.test_count() * k);
    return {static_cast<double>(within) / total, static_cast<double>(overlap) / total};
  }
};

void normalize_rows(std::vector<float>& rows, std::size_t dim) {
  const std::size_t n = rows.size() / dim;
  parallel_for(n, 8, [&](std::size_t i, std::size_t) {
    float* row = &rows[i * dim];
    const float norm = std::sqrt(kernels::dot_scalar(row, row, dim));
    if (norm > 0.0f) {
      for (std::size_t d = 0; d < dim; ++d) {
        row[d] /= norm;
      }
    }
  });
}

std::vector<float> prepared_rows(const float* source, std::size_t n, const Dataset& dataset) {
  std::vector<float> rows(source, source + n * dataset.dim());
  if (dataset.metric() == Metric::Angular) {
    normalize_rows(rows, dataset.dim());
  }
  return rows;
}

using Flags = std::map<std::string, std::string>;

Flags parse_flags(int argc, char** argv) {
  Flags flags;
  for (int i = 2; i + 1 < argc; i += 2) {
    flags[argv[i]] = argv[i + 1];
  }
  return flags;
}

const std::string& need(const Flags& flags, const std::string& name) {
  const auto it = flags.find(name);
  if (it == flags.end()) {
    throw std::invalid_argument("missing " + name);
  }
  return it->second;
}

std::size_t size_or(const Flags& flags, const std::string& name, std::size_t fallback) {
  const auto it = flags.find(name);
  return it == flags.end() ? fallback : std::stoull(it->second);
}

std::string dataset_name(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

std::filesystem::path index_path(const Flags& flags, const std::string& engine) {
  return std::filesystem::path(need(flags, "--index-dir")) /
         (dataset_name(need(flags, "--dataset")) + "-M" + need(flags, "--M") + "-efc" +
          need(flags, "--efc") + "." + engine);
}

EngineConfig config_for(const Dataset& dataset, const Flags& flags) {
  EngineConfig config;
  config.dim = dataset.dim();
  config.metric = dataset.metric();
  config.capacity = dataset.train_count();
  config.M = static_cast<std::uint32_t>(std::stoul(need(flags, "--M")));
  config.ef_construction = static_cast<std::uint32_t>(std::stoul(need(flags, "--efc")));
  return config;
}

// Rows are appended, so a header is written only to a new or empty file.
std::ofstream open_csv(const std::string& path, const std::string& header) {
  const bool fresh = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  if (fresh) {
    out << header << '\n';
  }
  return out;
}

int run_build(const Flags& flags) {
  const Dataset dataset = Dataset::open(need(flags, "--dataset"));
  const std::string engine_name = need(flags, "--engine");
  const std::size_t threads = size_or(flags, "--threads", 8);
  const std::vector<float> rows = prepared_rows(dataset.train(), dataset.train_count(), dataset);

  const std::unique_ptr<Engine> engine = make_engine(engine_name, config_for(dataset, flags));
  const double load_before = load_average();
  const auto start = Clock::now();
  engine->build(rows.data(), dataset.train_count(), threads);
  const double build_seconds = seconds_since(start);
  const double load_after = load_average();
  const std::size_t rss = peak_rss_bytes();

  const std::filesystem::path path = index_path(flags, engine_name);
  std::filesystem::create_directories(path.parent_path());
  const auto save_start = Clock::now();
  engine->save(path);
  const double save_seconds = seconds_since(save_start);

  std::ofstream out = open_csv(need(flags, "--out"),
                               "dataset,engine,M,efc,threads,build_seconds,save_seconds,"
                               "peak_rss_bytes,engine_bytes,file_bytes,load_before,load_after");
  out << dataset_name(need(flags, "--dataset")) << ',' << engine_name << ',' << need(flags, "--M")
      << ',' << need(flags, "--efc") << ',' << threads << ',' << build_seconds << ','
      << save_seconds << ',' << rss << ',' << engine->memory_bytes() << ','
      << std::filesystem::file_size(path) << ',' << load_before << ',' << load_after << '\n';
  std::printf("%-14s built in %7.2f s, peak RSS %.0f MiB, load %.1f -> %.1f\n", engine_name.c_str(),
              build_seconds, static_cast<double>(rss) / 1048576.0, load_before, load_after);
  return 0;
}

// Runs every query once and returns elapsed seconds.
double timed_pass(const Engine& engine, const float* queries, std::size_t dim, std::size_t count,
                  std::size_t k, std::size_t threads) {
  const auto start = Clock::now();
  parallel_for(
      count, threads,
      [&](std::size_t q, std::size_t) {
        thread_local std::vector<std::uint64_t> labels;
        labels.resize(k);
        engine.search(queries + q * dim, k, labels.data());
      },
      16);
  return seconds_since(start);
}

struct Cell {
  std::vector<double> qps;
  double recall = 0.0;
  double recall_ids = 0.0;
  double distance_computations = 0.0;
  double p50_us = 0.0;
  double p99_us = 0.0;
  double load_sum = 0.0;
};

int run_search(const Flags& flags) {
  const Dataset dataset = Dataset::open(need(flags, "--dataset"));
  const std::vector<std::string> names = split(need(flags, "--engines"));
  const std::vector<std::size_t> efs = split_sizes(need(flags, "--ef"));
  const std::size_t k = size_or(flags, "--k", 10);
  const std::size_t threads = size_or(flags, "--threads", 1);
  const std::size_t passes = size_or(flags, "--passes", 5);
  const std::size_t dim = dataset.dim();
  const std::size_t query_count = dataset.test_count();

  RecallScorer scorer{dataset, prepared_rows(dataset.train(), dataset.train_count(), dataset),
                      prepared_rows(dataset.test(), query_count, dataset), k};
  const float* queries = scorer.queries.data();

  std::vector<std::unique_ptr<Engine>> engines;
  std::map<std::string, double> load_seconds;
  const EngineConfig config = config_for(dataset, flags);
  for (const std::string& name : names) {
    engines.push_back(make_engine(name, config));
    const auto start = Clock::now();
    engines.back()->load(index_path(flags, name));
    load_seconds[name] = seconds_since(start);
    std::printf("%-14s load %.3f s\n", name.c_str(), load_seconds[name]);
  }

  // Warm every engine so the page cache and TLB state do not favour whichever ran last.
  for (auto& engine : engines) {
    engine->set_ef(*std::max_element(efs.begin(), efs.end()));
    for (std::size_t q = 0; q < std::min<std::size_t>(query_count, 1000); ++q) {
      std::uint64_t ignored[1024];
      engine->search(queries + q * dim, std::min<std::size_t>(k, 1024), ignored);
    }
  }

  std::vector<std::vector<Cell>> cells(engines.size(), std::vector<Cell>(efs.size()));

  // Untimed: recall and distance counts do not vary between passes for a fixed index.
  for (std::size_t e = 0; e < engines.size(); ++e) {
    for (std::size_t i = 0; i < efs.size(); ++i) {
      engines[e]->set_ef(efs[i]);
      std::vector<std::uint64_t> labels(query_count * k);
      std::vector<std::uint32_t> found(query_count);
      std::uint64_t computations = 0;
      for (std::size_t q = 0; q < query_count; ++q) {
        found[q] = static_cast<std::uint32_t>(
            engines[e]->search_counted(queries + q * dim, k, &labels[q * k], &computations));
      }
      const auto [recall, recall_ids] = scorer.score(labels, found);
      cells[e][i].recall = recall;
      cells[e][i].recall_ids = recall_ids;
      cells[e][i].distance_computations =
          static_cast<double>(computations) / static_cast<double>(query_count);
    }
  }

  for (std::size_t pass = 0; pass < passes; ++pass) {
    const double pass_start_load = load_average();
    for (std::size_t i = 0; i < efs.size(); ++i) {
      // Rotating the order removes any systematic "always runs after X" effect.
      for (std::size_t offset = 0; offset < engines.size(); ++offset) {
        const std::size_t e = (offset + pass + i) % engines.size();
        engines[e]->set_ef(efs[i]);
        const double seconds = timed_pass(*engines[e], queries, dim, query_count, k, threads);
        cells[e][i].qps.push_back(static_cast<double>(query_count) / seconds);
        cells[e][i].load_sum += load_average();
      }
    }
    std::printf("pass %zu/%zu done, load %.1f -> %.1f\n", pass + 1, passes, pass_start_load,
                load_average());
    std::fflush(stdout);
  }

  // Latency is a separate pass because per-query timers would otherwise sit in the
  // QPS measurement.
  for (std::size_t e = 0; e < engines.size(); ++e) {
    for (std::size_t i = 0; i < efs.size(); ++i) {
      engines[e]->set_ef(efs[i]);
      std::vector<double> micros(query_count);
      std::vector<std::uint64_t> labels(k);
      for (std::size_t q = 0; q < query_count; ++q) {
        const auto start = Clock::now();
        engines[e]->search(queries + q * dim, k, labels.data());
        micros[q] = seconds_since(start) * 1e6;
      }
      std::sort(micros.begin(), micros.end());
      cells[e][i].p50_us = micros[query_count / 2];
      cells[e][i].p99_us = micros[query_count * 99 / 100];
    }
  }

  std::ofstream out = open_csv(need(flags, "--out"),
                               "dataset,engine,M,efc,threads,k,ef,recall,recall_ids,qps_best,"
                               "qps_median,qps_worst,distance_computations,p50_us,p99_us,"
                               "mean_load,load_seconds");
  for (std::size_t e = 0; e < engines.size(); ++e) {
    for (std::size_t i = 0; i < efs.size(); ++i) {
      Cell& cell = cells[e][i];
      std::sort(cell.qps.begin(), cell.qps.end());
      out << dataset_name(need(flags, "--dataset")) << ',' << names[e] << ',' << need(flags, "--M")
          << ',' << need(flags, "--efc") << ',' << threads << ',' << k << ',' << efs[i] << ','
          << cell.recall << ',' << cell.recall_ids << ',' << cell.qps.back() << ','
          << cell.qps[cell.qps.size() / 2] << ',' << cell.qps.front() << ','
          << cell.distance_computations << ',' << cell.p50_us << ',' << cell.p99_us << ','
          << cell.load_sum / static_cast<double>(cell.qps.size()) << ',' << load_seconds[names[e]]
          << '\n';
    }
  }
  return 0;
}

int run_pq(const Flags& flags) {
  const Dataset dataset = Dataset::open(need(flags, "--dataset"));
  const std::size_t threads = size_or(flags, "--threads", 8);
  const std::size_t k = size_or(flags, "--k", 10);
  const std::size_t passes = size_or(flags, "--passes", 3);
  const std::size_t dim = dataset.dim();
  const std::size_t query_count = dataset.test_count();

  RecallScorer scorer{dataset, prepared_rows(dataset.train(), dataset.train_count(), dataset),
                      prepared_rows(dataset.test(), query_count, dataset), k};
  const float* queries = scorer.queries.data();

  IndexOptions options;
  options.dim = dim;
  options.metric = dataset.metric();
  options.capacity = dataset.train_count();
  options.M = static_cast<std::uint32_t>(std::stoul(need(flags, "--M")));
  options.ef_construction = static_cast<std::uint32_t>(std::stoul(need(flags, "--efc")));
  Index index(options);
  index.add_batch({scorer.rows.data(), scorer.rows.size()}, threads);

  std::ofstream out = open_csv(need(flags, "--out"),
                               "dataset,M,efc,pq_m,ef,rerank,recall,recall_ids,qps_best,"
                               "train_seconds,code_bytes,vector_bytes,link_bytes,mean_load");

  struct Setting {
    std::size_t ef;
    std::size_t rerank;
  };
  std::vector<Setting> settings;
  const std::vector<std::size_t> efs = split_sizes(need(flags, "--ef"));
  const std::vector<std::size_t> reranks = split_sizes(need(flags, "--rerank"));
  for (const std::size_t ef : efs) {
    for (const std::size_t rerank : reranks) {
      if (rerank <= ef) {
        settings.push_back({ef, rerank});
      }
    }
  }

  const auto evaluate = [&](std::size_t pq_m, const Setting& setting, double train_seconds) {
    std::vector<std::uint64_t> labels(query_count * k);
    std::vector<std::uint32_t> found(query_count);
    std::vector<Neighbor> scratch(k);
    const auto search_one = [&](std::size_t q, std::span<Neighbor> result) {
      return pq_m == 0
                 ? index.search({queries + q * dim, dim}, setting.ef, result)
                 : index.search_pq({queries + q * dim, dim}, setting.ef, setting.rerank, result);
    };
    for (std::size_t q = 0; q < query_count; ++q) {
      found[q] = static_cast<std::uint32_t>(search_one(q, scratch));
      for (std::size_t i = 0; i < found[q]; ++i) {
        labels[q * k + i] = scratch[i].label;
      }
    }
    double best_seconds = 1e30;
    double load = 0.0;
    for (std::size_t pass = 0; pass < passes; ++pass) {
      const auto start = Clock::now();
      for (std::size_t q = 0; q < query_count; ++q) {
        search_one(q, scratch);
      }
      best_seconds = std::min(best_seconds, seconds_since(start));
      load += load_average();
    }
    const auto [recall, recall_ids] = scorer.score(labels, found);
    const MemoryBreakdown memory = index.memory();
    out << dataset_name(need(flags, "--dataset")) << ',' << options.M << ','
        << options.ef_construction << ',' << pq_m << ',' << setting.ef << ',' << setting.rerank
        << ',' << recall << ',' << recall_ids << ','
        << static_cast<double>(query_count) / best_seconds << ',' << train_seconds << ','
        << memory.pq << ',' << memory.vectors << ',' << memory.links << ','
        << load / static_cast<double>(passes) << '\n';
    out.flush();
  };

  // m = 0 is the exact graph at the same ef, as the reference line for each sweep.
  for (const std::size_t ef : efs) {
    evaluate(0, {ef, 0}, 0.0);
  }
  for (const std::size_t pq_m : split_sizes(need(flags, "--pq-m"))) {
    PqTrainOptions pq;
    pq.m = pq_m;
    pq.threads = threads;
    const auto start = Clock::now();
    index.attach_pq(pq);
    const double train_seconds = seconds_since(start);
    std::printf("pq m=%zu trained and encoded in %.1f s\n", pq_m, train_seconds);
    std::fflush(stdout);
    for (const Setting& setting : settings) {
      evaluate(pq_m, setting, train_seconds);
    }
  }
  return 0;
}

// What opening an index really costs. mmap makes open() nearly free by deferring the
// page faults to the first queries, so open time alone would flatter it. This reports
// open plus the first queries, in a fresh process, with the page cache warm.
int run_first_queries(const Flags& flags) {
  const Dataset dataset = Dataset::open(need(flags, "--dataset"));
  const std::string engine_name = need(flags, "--engine");
  const std::size_t count = size_or(flags, "--count", 1000);
  const std::size_t dim = dataset.dim();
  const std::vector<float> queries = prepared_rows(dataset.test(), count, dataset);

  const std::unique_ptr<Engine> engine = make_engine(engine_name, config_for(dataset, flags));
  const auto load_start = Clock::now();
  engine->load(index_path(flags, engine_name));
  const double load_seconds = seconds_since(load_start);
  engine->set_ef(size_or(flags, "--ef", 100));

  std::uint64_t labels[10];
  const auto run_queries = [&] {
    const auto start = Clock::now();
    for (std::size_t q = 0; q < count; ++q) {
      engine->search(&queries[q * dim], 10, labels);
    }
    return seconds_since(start);
  };
  const double first = run_queries();
  const double second = run_queries();

  std::ofstream out = open_csv(need(flags, "--out"),
                               "dataset,engine,load_seconds,first_queries_seconds,"
                               "second_queries_seconds,queries,mean_load");
  out << dataset_name(need(flags, "--dataset")) << ',' << engine_name << ',' << load_seconds << ','
      << first << ',' << second << ',' << count << ',' << load_average() << '\n';
  return 0;
}

// Isolates prefetch depth: same index, same process, only the line count differs. The
// settings are interleaved per pass for the same reason engines are in run_search.
int run_prefetch(const Flags& flags) {
  const Dataset dataset = Dataset::open(need(flags, "--dataset"));
  Index index = Index::open(index_path(flags, "annvil"));
  const std::vector<std::size_t> depths = split_sizes(need(flags, "--lines"));
  const std::vector<std::size_t> efs = split_sizes(need(flags, "--ef"));
  const std::size_t passes = size_or(flags, "--passes", 7);
  const std::size_t dim = dataset.dim();
  const std::size_t query_count = dataset.test_count();
  const std::vector<float> queries = prepared_rows(dataset.test(), query_count, dataset);

  std::vector<Neighbor> out(10);
  const auto run_queries = [&](std::size_t ef) {
    for (std::size_t q = 0; q < query_count; ++q) {
      index.search({&queries[q * dim], dim}, ef, out);
    }
  };
  run_queries(efs.back());

  std::map<std::pair<std::size_t, std::size_t>, std::vector<double>> qps;
  double load = 0.0;
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (const std::size_t ef : efs) {
      for (std::size_t offset = 0; offset < depths.size(); ++offset) {
        const std::size_t lines = depths[(offset + pass) % depths.size()];
        index.set_prefetch_lines(lines);
        const auto start = Clock::now();
        run_queries(ef);
        qps[{lines, ef}].push_back(static_cast<double>(query_count) / seconds_since(start));
      }
    }
    load += load_average();
  }

  std::ofstream csv =
      open_csv(need(flags, "--out"), "dataset,M,lines,ef,qps_best,qps_median,mean_load");
  for (auto& [key, samples] : qps) {
    std::sort(samples.begin(), samples.end());
    csv << dataset_name(need(flags, "--dataset")) << ',' << need(flags, "--M") << ',' << key.first
        << ',' << key.second << ',' << samples.back() << ',' << samples[samples.size() / 2] << ','
        << load / static_cast<double>(passes) << '\n';
  }
  return 0;
}

// Nanoseconds per distance on data that fits in L1, so memory is out of the picture
// and only the arithmetic is compared.
int run_kernels(const Flags& flags) {
  constexpr std::size_t kWorkingSetBytes = 48 * 1024;
  const std::size_t calls_per_attempt = size_or(flags, "--calls", 20000000);
  std::ofstream out = open_csv(need(flags, "--out"), "kernel,dim,metric,ns_per_call,mean_load");

  for (const std::size_t dim : split_sizes(need(flags, "--dims"))) {
    const std::size_t pairs =
        std::max<std::size_t>(1, kWorkingSetBytes / (2 * dim * sizeof(float)));
    const std::size_t rounds = calls_per_attempt / pairs;
    std::vector<float> data(2 * pairs * dim);
    std::uint32_t state = 12345;
    for (float& x : data) {
      state = state * 1664525u + 1013904223u;
      x = static_cast<float>(state >> 8) * 1e-6f;
    }
    const auto measure = [&](const std::string& name, const std::string& metric, auto&& call) {
      double best = 1e30;
      for (int attempt = 0; attempt < 9; ++attempt) {
        float sink = 0.0f;
        const auto start = Clock::now();
        for (std::size_t r = 0; r < rounds; ++r) {
          for (std::size_t p = 0; p < pairs; ++p) {
            sink += call(&data[2 * p * dim], &data[(2 * p + 1) * dim]);
          }
        }
        const double elapsed = seconds_since(start);
        // Keeps the loop from being optimized away.
        if (sink == 1.0f) {
          std::puts("");
        }
        best = std::min(best, elapsed);
      }
      const double ns = best * 1e9 / static_cast<double>(rounds * pairs);
      out << name << ',' << dim << ',' << metric << ',' << ns << ',' << load_average() << '\n';
      std::printf("%-16s dim %4zu %-4s %7.2f ns\n", name.c_str(), dim, metric.c_str(), ns);
    };

    const auto l2 = hnswlib_raw_kernel(dim, Metric::L2);
    const auto ip = hnswlib_raw_kernel(dim, Metric::Angular);
    measure("annvil-simd", "l2",
            [&](const float* a, const float* b) { return kernels::l2_sq(a, b, dim); });
    measure("scalar", "l2",
            [&](const float* a, const float* b) { return kernels::l2_sq_scalar(a, b, dim); });
    measure("hnswlib", "l2",
            [&](const float* a, const float* b) { return l2.function(a, b, l2.param); });
    measure("annvil-simd", "dot",
            [&](const float* a, const float* b) { return 1.0f - kernels::dot(a, b, dim); });
    measure("scalar", "dot",
            [&](const float* a, const float* b) { return 1.0f - kernels::dot_scalar(a, b, dim); });
    measure("hnswlib", "dot",
            [&](const float* a, const float* b) { return ip.function(a, b, ip.param); });
  }
  return 0;
}

int run_info() {
  std::printf("{\"kernel\":\"%s\",\"annvil_flags\":\"%s\",\"hnswlib_flags\":\"%s\"}\n",
              kernels::active_kernel(), ANNVIL_BENCH_ANNVIL_FLAGS, ANNVIL_BENCH_HNSWLIB_FLAGS);
  return 0;
}

}  // namespace
}  // namespace annvil::bench

int main(int argc, char** argv) {
  using namespace annvil::bench;
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    const Flags flags = parse_flags(argc, argv);
    if (mode == "build") {
      return run_build(flags);
    }
    if (mode == "search") {
      return run_search(flags);
    }
    if (mode == "pq") {
      return run_pq(flags);
    }
    if (mode == "kernels") {
      return run_kernels(flags);
    }
    if (mode == "prefetch") {
      return run_prefetch(flags);
    }
    if (mode == "first-queries") {
      return run_first_queries(flags);
    }
    if (mode == "info") {
      return run_info();
    }
    std::fputs(
        "usage: annvil_bench build|search|pq|kernels|prefetch|first-queries|info [--flag value "
        "...]\n",
        stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
