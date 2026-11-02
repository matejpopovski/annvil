// How much does building in parallel cost in graph quality, and what does the linking
// order have to do with it? Repeats whole builds on one small clustered data set, since
// a single parallel build is a poor sample. Linked twice: against the normal library,
// and against bench/top_down_linking.patch applied to src/index.cpp.
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/parallel.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 16;
constexpr std::size_t kCount = 4000;
constexpr std::size_t kK = 10;
constexpr std::size_t kQueries = 50;
constexpr std::size_t kEf = 100;
constexpr std::uint32_t kM = 8;
constexpr std::uint32_t kEfConstruction = 48;

struct Samples {
  std::vector<double> recall;
  std::vector<std::size_t> unreachable;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s LABEL BUILDS OUT.csv\n", argv[0]);
    return 2;
  }
  const std::string label = argv[1];
  const int builds = std::stoi(argv[2]);
  const auto data = clustered(kCount, kDim, 41);
  const auto queries = clustered(kQueries, kDim, 42);

  std::vector<std::set<std::uint64_t>> truth;
  for (std::size_t q = 0; q < kQueries; ++q) {
    const auto labels = exact_knn(data, kDim, &queries[q * kDim], kK, Metric::L2);
    truth.emplace_back(labels.begin(), labels.end());
  }

  std::ofstream out(argv[3], std::ios::app);
  const auto report = [&](const std::string& engine, std::size_t threads, const Samples& s) {
    double sum = 0.0;
    for (const double r : s.recall) {
      sum += r;
    }
    const auto [low, high] = std::minmax_element(s.recall.begin(), s.recall.end());
    std::size_t worst = 0;
    std::size_t affected = 0;
    for (const std::size_t u : s.unreachable) {
      worst = std::max(worst, u);
      affected += u > 0 ? 1 : 0;
    }
    const bool measured = !s.unreachable.empty();
    out << engine << ',' << threads << ',' << s.recall.size() << ','
        << sum / static_cast<double>(s.recall.size()) << ',' << *low << ',' << *high << ','
        << (measured ? std::to_string(affected) : "") << ','
        << (measured ? std::to_string(worst) : "") << '\n';
    std::printf("%-22s threads=%zu mean %.4f min %.3f max %.3f unreachable builds %zu worst %zu\n",
                engine.c_str(), threads, sum / static_cast<double>(s.recall.size()), *low, *high,
                affected, worst);
  };

  for (const std::size_t threads : {std::size_t{1}, std::size_t{8}}) {
    // A serial build is deterministic, so repeating it would add no information.
    const int repetitions = threads == 1 ? 1 : builds;
    Samples annvil_samples, hnswlib_samples;
    for (int build = 0; build < repetitions; ++build) {
      IndexOptions options;
      options.dim = kDim;
      options.capacity = kCount;
      options.M = kM;
      options.ef_construction = kEfConstruction;
      Index index(options);
      index.add_batch(data, threads);
      std::size_t hits = 0;
      std::vector<Neighbor> found(kK);
      for (std::size_t q = 0; q < kQueries; ++q) {
        const std::size_t count = index.search({&queries[q * kDim], kDim}, kEf, found);
        for (std::size_t i = 0; i < count; ++i) {
          hits += truth[q].count(found[i].label);
        }
      }
      annvil_samples.recall.push_back(static_cast<double>(hits) / (kQueries * kK));
      annvil_samples.unreachable.push_back(unreachable_nodes(index));

      if (label == "bottom-up") {
        hnswlib::L2Space space(kDim);
        hnswlib::HierarchicalNSW<float> baseline(&space, kCount, kM, kEfConstruction);
        parallel_for(kCount, threads,
                     [&](std::size_t i, std::size_t) { baseline.addPoint(&data[i * kDim], i); });
        baseline.setEf(kEf);
        hits = 0;
        for (std::size_t q = 0; q < kQueries; ++q) {
          auto heap = baseline.searchKnn(&queries[q * kDim], kK);
          for (; !heap.empty(); heap.pop()) {
            hits += truth[q].count(heap.top().second);
          }
        }
        hnswlib_samples.recall.push_back(static_cast<double>(hits) / (kQueries * kK));
      }
    }
    report("annvil-" + label, threads, annvil_samples);
    if (label == "bottom-up") {
      report("hnswlib", threads, hnswlib_samples);
    }
  }
  return 0;
}
