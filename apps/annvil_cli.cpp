// Command-line front end: build an index from a .annb dataset, inspect it, or serve it.
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "annvil/dataset.hpp"
#include "annvil/index.hpp"
#include "annvil/kernels.hpp"
#include "server.hpp"

namespace {

using annvil::Dataset;
using annvil::Index;

constexpr const char* kUsage = R"(usage:
  annvil build  --dataset FILE.annb --out FILE [--M 16] [--ef-construction 200]
                [--threads N] [--reorder] [--pq-m M] [--seed S]
  annvil info   FILE
  annvil verify FILE
  annvil recall --index FILE --dataset FILE.annb [--k 10] [--ef 100]
  annvil serve  --index FILE [--host 127.0.0.1] [--port 8080] [--threads 4]
)";

// Maps "--name value" pairs, and bare "--name" switches to "1".
std::map<std::string, std::string> parse_flags(int argc, char** argv, int first) {
  static const std::set<std::string> kSwitches = {"--reorder"};
  std::map<std::string, std::string> flags;
  for (int i = first; i < argc; ++i) {
    const std::string name = argv[i];
    if (name.rfind("--", 0) != 0) {
      throw std::invalid_argument("unexpected argument: " + name);
    }
    if (kSwitches.count(name) != 0) {
      flags[name] = "1";
    } else {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
      }
      flags[name] = argv[++i];
    }
  }
  return flags;
}

const std::string& required(const std::map<std::string, std::string>& flags,
                            const std::string& name) {
  const auto it = flags.find(name);
  if (it == flags.end()) {
    throw std::invalid_argument("missing required flag " + name);
  }
  return it->second;
}

std::size_t size_flag(const std::map<std::string, std::string>& flags, const std::string& name,
                      std::size_t fallback) {
  const auto it = flags.find(name);
  return it == flags.end() ? fallback : std::stoull(it->second);
}

std::size_t default_threads() {
  return std::max(1u, std::thread::hardware_concurrency());
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int cmd_build(const std::map<std::string, std::string>& flags) {
  const Dataset dataset = Dataset::open(required(flags, "--dataset"));
  annvil::IndexOptions options;
  options.dim = dataset.dim();
  options.metric = dataset.metric();
  options.capacity = dataset.train_count();
  options.M = static_cast<std::uint32_t>(size_flag(flags, "--M", 16));
  options.ef_construction = static_cast<std::uint32_t>(size_flag(flags, "--ef-construction", 200));
  options.seed = size_flag(flags, "--seed", 42);
  const std::size_t threads = size_flag(flags, "--threads", default_threads());

  Index index(options);
  const auto start = std::chrono::steady_clock::now();
  index.add_batch({dataset.train(), dataset.train_count() * dataset.dim()}, threads);
  std::printf("built %zu points in %.2f s on %zu threads\n", index.size(), elapsed_seconds(start),
              threads);
  if (flags.count("--reorder") != 0) {
    const auto reorder_start = std::chrono::steady_clock::now();
    index.reorder();
    std::printf("reordered in %.2f s\n", elapsed_seconds(reorder_start));
  }
  if (const std::size_t pq_m = size_flag(flags, "--pq-m", 0); pq_m != 0) {
    annvil::PqTrainOptions pq;
    pq.m = pq_m;
    pq.threads = threads;
    const auto pq_start = std::chrono::steady_clock::now();
    index.attach_pq(pq);
    std::printf("trained PQ m=%zu in %.2f s\n", pq_m, elapsed_seconds(pq_start));
  }
  index.save(required(flags, "--out"));
  return 0;
}

int cmd_info(const std::string& path) {
  const Index index = Index::open(path);
  const annvil::MemoryBreakdown memory = index.memory();
  std::printf("size             %zu\n", index.size());
  std::printf("dim              %zu\n", index.dim());
  std::printf("metric           %s\n", index.metric() == annvil::Metric::L2 ? "l2" : "angular");
  std::printf("M                %u\n", index.M());
  std::printf("ef_construction  %u\n", index.ef_construction());
  std::printf("max level        %d\n", index.max_level());
  std::printf("pq m             %zu\n", index.pq_m());
  std::printf("kernel           %s\n", annvil::kernels::active_kernel());
  std::printf("vectors          %.1f MiB\n", static_cast<double>(memory.vectors) / 1048576.0);
  std::printf("links            %.1f MiB\n", static_cast<double>(memory.links) / 1048576.0);
  std::printf("labels           %.1f MiB\n", static_cast<double>(memory.labels) / 1048576.0);
  std::printf("pq               %.1f MiB\n", static_cast<double>(memory.pq) / 1048576.0);
  return 0;
}

int cmd_verify(const std::string& path) {
  const Index index = Index::open(path);
  const annvil::VerifyReport report = index.verify();
  for (const std::string& error : report.errors) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
  }
  std::printf("%s: %s\n", path.c_str(), report.ok ? "ok" : "CORRUPT");
  return report.ok ? 0 : 1;
}

int cmd_recall(const std::map<std::string, std::string>& flags) {
  const Index index = Index::open(required(flags, "--index"));
  const Dataset dataset = Dataset::open(required(flags, "--dataset"));
  const std::size_t k = size_flag(flags, "--k", 10);
  const std::size_t ef = size_flag(flags, "--ef", 100);
  if (k > dataset.gt_k()) {
    throw std::invalid_argument("dataset has too few ground-truth neighbours for this k");
  }
  std::vector<annvil::Neighbor> out(k);
  std::size_t hits = 0;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t q = 0; q < dataset.test_count(); ++q) {
    const std::size_t found = index.search({dataset.test_vector(q), dataset.dim()}, ef, out);
    const std::int32_t* truth = dataset.gt_neighbors(q);
    for (std::size_t i = 0; i < found; ++i) {
      hits += std::find(truth, truth + k, static_cast<std::int32_t>(out[i].label)) != truth + k;
    }
  }
  const double seconds = elapsed_seconds(start);
  std::printf("k=%zu ef=%zu recall=%.4f qps=%.0f (1 thread)\n", k, ef,
              static_cast<double>(hits) / static_cast<double>(dataset.test_count() * k),
              static_cast<double>(dataset.test_count()) / seconds);
  return 0;
}

int cmd_serve(const std::map<std::string, std::string>& flags) {
  // A server should not start on a file it would later fault on.
  const Index index = Index::open(required(flags, "--index"), /*verify=*/true);
  annvil::ServerOptions options;
  if (flags.count("--host") != 0) {
    options.host = flags.at("--host");
  }
  options.port = static_cast<int>(size_flag(flags, "--port", 8080));
  options.threads = size_flag(flags, "--threads", 4);
  annvil::Server server(index, options);
  const int port = server.bind();
  // The index is a read-only mapping and there is no state to flush, so exiting from
  // the handler loses nothing.
  std::signal(SIGINT, [](int) { std::_Exit(0); });
  std::signal(SIGTERM, [](int) { std::_Exit(0); });
  std::printf("serving %zu vectors on http://%s:%d\n", index.size(), options.host.c_str(), port);
  std::fflush(stdout);
  server.run();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fputs(kUsage, stderr);
      return 2;
    }
    const std::string command = argv[1];
    if (command == "info" && argc == 3) {
      return cmd_info(argv[2]);
    }
    if (command == "verify" && argc == 3) {
      return cmd_verify(argv[2]);
    }
    const auto flags = parse_flags(argc, argv, 2);
    if (command == "build") {
      return cmd_build(flags);
    }
    if (command == "recall") {
      return cmd_recall(flags);
    }
    if (command == "serve") {
      return cmd_serve(flags);
    }
    std::fputs(kUsage, stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
