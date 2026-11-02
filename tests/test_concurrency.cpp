// These tests exist mainly to be run under ThreadSanitizer: the assertions catch lost
// updates, and TSan catches the races that happen to produce correct answers.
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/parallel.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 16;

// GCC defines __SANITIZE_THREAD__; Clang reports it through __has_feature, which GCC 13
// cannot even parse, so the two checks must not share an #if.
#if defined(__SANITIZE_THREAD__)
#define ANNVIL_UNDER_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define ANNVIL_UNDER_TSAN 1
#endif
#endif

#if defined(ANNVIL_UNDER_TSAN)
constexpr std::size_t kScale = 1;
#else
constexpr std::size_t kScale = 4;
#endif

IndexOptions options(std::size_t capacity) {
  IndexOptions o;
  o.dim = kDim;
  o.capacity = capacity;
  o.M = 8;
  o.ef_construction = 48;
  return o;
}

}  // namespace

TEST_CASE("parallel_for visits every index exactly once") {
  std::vector<std::atomic<int>> hits(10000);
  parallel_for(10000, 8, [&](std::size_t i, std::size_t) { hits[i].fetch_add(1); });
  for (const auto& h : hits) {
    CHECK(h.load() == 1);
  }
}

TEST_CASE("parallel_for propagates the first exception") {
  CHECK_THROWS_AS(parallel_for(10000, 4,
                               [&](std::size_t i, std::size_t) {
                                 if (i == 1234) {
                                   throw std::runtime_error("boom");
                                 }
                               }),
                  std::runtime_error);
}

// One parallel build is a poor sample: recall on this data varies from 0.92 to 0.98
// between runs, in hnswlib as much as in annvil, because concurrent inserts cannot see
// each other (see results/parallel_quality.csv). So these tests average several builds.
TEST_CASE("parallel builds stay structurally valid and close to serial recall") {
  const std::size_t n = 1000 * kScale;
  const auto data = clustered(n, kDim, 41);
  const auto queries = clustered(50, kDim, 42);
  const auto recall = [&](const Index& index) {
    return mean_recall(data, queries, kDim, 10, Metric::L2,
                       [&](std::span<const float> q, std::span<Neighbor> out) {
                         return index.search(q, 100, out);
                       });
  };

  Index serial(options(n));
  serial.add_batch(data, 1);

  constexpr int kBuilds = 5;
  double total = 0.0;
  for (int build = 0; build < kBuilds; ++build) {
    Index parallel(options(n));
    parallel.add_batch(data, 8);
    CHECK(parallel.size() == n);
    CHECK(parallel.verify().ok);
    total += recall(parallel);
  }
  CHECK(total / kBuilds >= recall(serial) - 0.05);
}

// Guards the bottom-up linking order. Linking top-down leaves a node visible on a level
// before the level below exists, which orphans dozens of nodes per build on this data.
TEST_CASE("a parallel build leaves almost every node reachable") {
  const std::size_t n = 1000 * kScale;
  const auto data = clustered(n, kDim, 41);
  std::size_t worst = 0;
  for (int build = 0; build < 8; ++build) {
    Index index(options(n));
    index.add_batch(data, 8);
    worst = std::max(worst, unreachable_nodes(index));
  }
  CHECK(worst * 400 <= n);
}

TEST_CASE("every label survives a contended build") {
  const std::size_t n = 600 * kScale;
  const auto data = clustered(n, kDim, 43);
  Index index(options(n));
  index.add_batch(data, 8);
  std::set<std::uint64_t> labels;
  std::vector<Neighbor> out(1);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(index.search(std::span<const float>(data).subspan(i * kDim, kDim), 64, out) == 1);
    labels.insert(out[0].label);
  }
  // Exact self-recall is not guaranteed, but it should be nearly complete.
  CHECK(static_cast<double>(labels.size()) >= 0.98 * static_cast<double>(n));
}

TEST_CASE("concurrent readers of one mapped index agree with a serial run") {
  TempDir dir;
  const std::size_t n = 2000;
  const auto data = clustered(n, kDim, 44);
  const auto queries = clustered(64, kDim, 45);
  Index built(options(n));
  built.add_batch(data, 4);
  built.save(dir.file("c.annvil"));
  const Index mapped = Index::open(dir.file("c.annvil"));

  std::vector<std::vector<Neighbor>> expected(64, std::vector<Neighbor>(10));
  for (std::size_t q = 0; q < 64; ++q) {
    mapped.search(std::span<const float>(queries).subspan(q * kDim, kDim), 64, expected[q]);
  }

  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      std::vector<Neighbor> out(10);
      for (std::size_t round = 0; round < 50 * kScale; ++round) {
        const std::size_t q = (round + static_cast<std::size_t>(t) * 13) % 64;
        mapped.search(std::span<const float>(queries).subspan(q * kDim, kDim), 64, out);
        for (std::size_t i = 0; i < 10; ++i) {
          if (out[i].label != expected[q][i].label) {
            mismatches.fetch_add(1);
          }
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(mismatches.load() == 0);
}

TEST_CASE("readers run safely and make progress during inserts") {
  const std::size_t n = 1000 * kScale;
  const auto data = clustered(n, kDim, 46);
  Index index(options(n));
  index.add_batch(std::span<const float>(data).subspan(0, 100 * kDim), 1);

  std::atomic<bool> done{false};
  std::atomic<std::size_t> searches{0};
  std::atomic<int> errors{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&, t] {
      std::vector<Neighbor> out(10);
      std::size_t q = static_cast<std::size_t>(t);
      while (!done.load(std::memory_order_acquire)) {
        const auto query = std::span<const float>(data).subspan((q++ % n) * kDim, kDim);
        const std::size_t found = index.search(query, 32, out);
        if (found == 0) {
          errors.fetch_add(1);
        }
        for (std::size_t i = 0; i < found; ++i) {
          if (out[i].label >= n) {
            errors.fetch_add(1);
          }
        }
        searches.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  parallel_for(n - 100, 4, [&](std::size_t i, std::size_t) {
    const std::size_t row = i + 100;
    index.add(std::span<const float>(data).subspan(row * kDim, kDim), row);
  });
  done.store(true, std::memory_order_release);
  for (auto& thread : readers) {
    thread.join();
  }

  CHECK(errors.load() == 0);
  CHECK(searches.load() > 0);
  CHECK(index.size() == n);
  CHECK(index.verify().ok);
}
