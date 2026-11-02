#include <doctest/doctest.h>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

#include "annvil/index.hpp"
#include "detail/search_state.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 32;
constexpr std::size_t kCount = 4000;

IndexOptions options_for(Metric metric, std::size_t capacity = kCount) {
  IndexOptions o;
  o.dim = kDim;
  o.metric = metric;
  o.capacity = capacity;
  o.M = 12;
  o.ef_construction = 100;
  return o;
}

Index build(const std::vector<float>& data, Metric metric, std::size_t threads = 1) {
  Index index(options_for(metric, data.size() / kDim));
  index.add_batch(data, threads);
  return index;
}

double recall_at(const Index& index, const std::vector<float>& data,
                 const std::vector<float>& queries, std::size_t k, std::size_t ef, Metric metric) {
  return mean_recall(
      data, queries, kDim, k, metric,
      [&](std::span<const float> q, std::span<Neighbor> out) { return index.search(q, ef, out); });
}

}  // namespace

TEST_CASE("the beam keeps the ef best entries sorted") {
  detail::Beam beam;
  beam.reset(4);
  CHECK(beam.bound() == std::numeric_limits<float>::max());
  for (const auto& [d, id] : std::vector<std::pair<float, std::uint32_t>>{
           {5.0f, 5}, {1.0f, 1}, {3.0f, 3}, {4.0f, 4}, {2.0f, 2}, {9.0f, 9}}) {
    if (d < beam.bound()) {
      beam.insert(d, id);
    }
  }
  REQUIRE(beam.size() == 4);
  for (std::uint32_t i = 0; i < 4; ++i) {
    CHECK(beam.id(i) == i + 1);
  }
  CHECK(beam.bound() == doctest::Approx(4.0f));
}

TEST_CASE("the beam expands closest-first and revisits the front after an insert") {
  detail::Beam beam;
  beam.reset(8);
  beam.insert(2.0f, 2);
  beam.insert(4.0f, 4);
  CHECK(beam.expand_next() == 2);
  beam.insert(1.0f, 1);
  CHECK(beam.peek_unexpanded() == 1);
  CHECK(beam.expand_next() == 1);
  CHECK(beam.expand_next() == 4);
  CHECK_FALSE(beam.has_unexpanded());
  CHECK(beam.id(1) == 2);  // The expanded tag must not leak into ids.
}

TEST_CASE("the beam handles an insert into the last slot of a full array") {
  detail::Beam beam;
  beam.reset(1);
  beam.insert(5.0f, 5);
  beam.insert(1.0f, 1);
  CHECK(beam.size() == 1);
  CHECK(beam.id(0) == 1);
}

TEST_CASE("the visited set resets between queries, including after growth") {
  detail::VisitedSet visited;
  visited.prepare(100);
  CHECK(visited.insert(7));
  CHECK_FALSE(visited.insert(7));
  CHECK(visited.insert(70));
  visited.prepare(100);
  CHECK(visited.insert(7));
  CHECK(visited.insert(70));
  visited.prepare(100000);
  CHECK(visited.insert(7));
  CHECK(visited.insert(99999));
}

TEST_CASE("recall on clustered data is high and does not fall as ef grows") {
  for (const Metric metric : {Metric::L2, Metric::Angular}) {
    CAPTURE(static_cast<int>(metric));
    auto data = clustered(kCount, kDim, 1);
    auto queries = clustered(200, kDim, 2);
    if (metric == Metric::Angular) {
      normalize_rows(data, kDim);
      normalize_rows(queries, kDim);
    }
    const Index index = build(data, metric);
    const double low = recall_at(index, data, queries, 10, 10, metric);
    const double mid = recall_at(index, data, queries, 10, 50, metric);
    const double high = recall_at(index, data, queries, 10, 400, metric);
    CHECK(mid >= low);
    CHECK(high >= mid);
    CHECK(high > 0.99);
  }
}

TEST_CASE("exact ties and duplicates do not break search or linking") {
  const std::vector<float> row(kDim, 1.5f);
  Index index(options_for(Metric::L2, 200));
  for (std::uint64_t i = 0; i < 200; ++i) {
    index.add(row, i);
  }
  std::vector<Neighbor> out(10);
  CHECK(index.search(row, 50, out) == 10);
  for (const Neighbor& n : out) {
    CHECK(n.distance == 0.0f);
  }
  std::set<std::uint64_t> labels;
  for (const Neighbor& n : out) {
    labels.insert(n.label);
  }
  CHECK(labels.size() == 10);
}

TEST_CASE("the graph respects its structural invariants") {
  const auto data = clustered(kCount, kDim, 3);
  const Index index = build(data, Metric::L2);
  const std::size_t M = index.M();
  std::size_t highest = 0;
  for (std::uint32_t id = 0; id < index.size(); ++id) {
    const int level = index.node_level(id);
    highest = std::max(highest, static_cast<std::size_t>(level));
    for (int lc = 0; lc <= level; ++lc) {
      const auto links = index.neighbors(id, lc);
      CHECK(links.size() <= (lc == 0 ? 2 * M : M));
      for (const std::uint32_t neighbor : links) {
        CHECK(neighbor != id);
        CHECK(neighbor < index.size());
        CHECK(index.node_level(neighbor) >= lc);
      }
      std::set<std::uint32_t> unique(links.begin(), links.end());
      CHECK(unique.size() == links.size());
    }
  }
  CHECK(static_cast<int>(highest) == index.max_level());
  const VerifyReport report = index.verify();
  CHECK(report.ok);
}

TEST_CASE("level sizes follow the exponential decay the paper prescribes") {
  const auto data = clustered(kCount, kDim, 4);
  const Index index = build(data, Metric::L2);
  std::map<int, std::size_t> at_level;
  for (std::uint32_t id = 0; id < index.size(); ++id) {
    ++at_level[index.node_level(id)];
  }
  // P(level >= 1) = 1 / M.
  const double share = 1.0 - static_cast<double>(at_level[0]) / static_cast<double>(index.size());
  CHECK(share == doctest::Approx(1.0 / index.M()).epsilon(0.25));
}

TEST_CASE("layer 0 stays connected") {
  const auto data = clustered(kCount, kDim, 5);
  const Index index = build(data, Metric::L2);
  std::vector<char> seen(index.size(), 0);
  std::vector<std::uint32_t> stack = {0};
  seen[0] = 1;
  std::size_t reached = 1;
  while (!stack.empty()) {
    const std::uint32_t node = stack.back();
    stack.pop_back();
    for (const std::uint32_t next : index.neighbors(node, 0)) {
      if (seen[next] == 0) {
        seen[next] = 1;
        ++reached;
        stack.push_back(next);
      }
    }
  }
  CHECK(static_cast<double>(reached) >= 0.99 * static_cast<double>(index.size()));
}

TEST_CASE("a single-threaded build is deterministic") {
  const auto data = clustered(1000, kDim, 6);
  const Index a = build(data, Metric::L2);
  const Index b = build(data, Metric::L2);
  for (std::uint32_t id = 0; id < a.size(); ++id) {
    CHECK(a.neighbors(id, 0) == b.neighbors(id, 0));
  }
}

TEST_CASE("custom labels come back from search") {
  const auto data = clustered(500, kDim, 7);
  std::vector<std::uint64_t> labels(500);
  std::iota(labels.begin(), labels.end(), std::uint64_t{1} << 40);
  Index index(options_for(Metric::L2, 500));
  index.add_batch(data, 1, labels);
  std::vector<Neighbor> out(1);
  REQUIRE(index.search(std::span<const float>(data).subspan(0, kDim), 100, out) == 1);
  CHECK(out[0].label == (std::uint64_t{1} << 40));
  CHECK(out[0].distance == doctest::Approx(0.0f));
}

TEST_CASE("angular distance ignores vector magnitude") {
  auto data = clustered(500, kDim, 8);
  Index index(options_for(Metric::Angular, 500));
  index.add_batch(data, 1);
  std::vector<float> scaled(data.begin(), data.begin() + kDim);
  for (float& x : scaled) {
    x *= 1000.0f;
  }
  std::vector<Neighbor> out(1);
  REQUIRE(index.search(scaled, 100, out) == 1);
  CHECK(out[0].label == 0);
  CHECK(out[0].distance == doctest::Approx(0.0f).epsilon(1e-5));
}

TEST_CASE("a zero vector under the angular metric stays finite") {
  Index index(options_for(Metric::Angular, 3));
  const std::vector<float> zero(kDim, 0.0f);
  std::vector<float> unit(kDim, 0.0f);
  unit[0] = 1.0f;
  index.add(zero, 0);
  index.add(unit, 1);
  std::vector<Neighbor> out(2);
  REQUIRE(index.search(zero, 10, out) == 2);
  CHECK(std::isfinite(out[0].distance));
  CHECK(std::isfinite(out[1].distance));
}

TEST_CASE("edge cases of search") {
  Index index(options_for(Metric::L2, 10));
  const std::vector<float> row(kDim, 0.5f);
  std::vector<Neighbor> out(5);

  SUBCASE("an empty index returns nothing") {
    CHECK(index.search(row, 10, out) == 0);
  }

  index.add(row, 7);

  SUBCASE("asking for more than exists returns what exists") {
    CHECK(index.search(row, 10, out) == 1);
    CHECK(out[0].label == 7);
  }
  SUBCASE("k of zero writes nothing") {
    CHECK(index.search(row, 10, std::span<Neighbor>()) == 0);
  }
  SUBCASE("ef below k is raised to k") {
    for (std::uint64_t i = 0; i < 9; ++i) {
      index.add(row, i);
    }
    CHECK(index.search(row, 1, out) == 5);
  }
  SUBCASE("a wrong-sized query is rejected") {
    const std::vector<float> wrong(kDim + 1, 0.0f);
    CHECK_THROWS_AS(index.search(wrong, 10, out), std::invalid_argument);
  }
  SUBCASE("a non-finite query is rejected") {
    std::vector<float> bad(kDim, 0.0f);
    bad[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_AS(index.search(bad, 10, out), std::invalid_argument);
  }
}

TEST_CASE("invalid construction and inserts are rejected") {
  IndexOptions o = options_for(Metric::L2, 10);
  o.dim = 0;
  CHECK_THROWS_AS(Index{o}, std::invalid_argument);
  o = options_for(Metric::L2, 10);
  o.M = 1;
  CHECK_THROWS_AS(Index{o}, std::invalid_argument);
  o = options_for(Metric::L2, 0);
  CHECK_THROWS_AS(Index{o}, std::invalid_argument);

  Index index(options_for(Metric::L2, 2));
  std::vector<float> row(kDim, 0.0f);
  CHECK_THROWS_AS(index.add(std::span<const float>(row).subspan(1), 0), std::invalid_argument);
  row[0] = std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(index.add(row, 0), std::invalid_argument);
  CHECK(index.size() == 0);
  row[0] = 1.0f;
  index.add(row, 0);
  index.add(row, 1);
  CHECK_THROWS_AS(index.add(row, 2), std::length_error);
  CHECK(index.size() == 2);
  CHECK_THROWS_AS(index.add_batch(std::span<const float>(row).subspan(1), 1),
                  std::invalid_argument);
}

TEST_CASE("search statistics count distance computations and grow with ef") {
  const auto data = clustered(kCount, kDim, 9);
  const Index index = build(data, Metric::L2);
  std::vector<Neighbor> out(10);
  SearchStats small, large;
  const std::span<const float> query = std::span<const float>(data).subspan(0, kDim);
  index.search(query, 10, out, &small);
  index.search(query, 200, out, &large);
  CHECK(small.distance_computations > 0);
  CHECK(large.distance_computations > small.distance_computations);
  CHECK(large.distance_computations <= index.size());
}

TEST_CASE("keep_pruned fills lists the heuristic left short") {
  const auto data = clustered(2000, kDim, 10);
  IndexOptions plain = options_for(Metric::L2, 2000);
  IndexOptions kept = plain;
  kept.keep_pruned = true;
  Index a(plain), b(kept);
  a.add_batch(data, 1);
  b.add_batch(data, 1);
  std::size_t links_a = 0, links_b = 0;
  for (std::uint32_t id = 0; id < 2000; ++id) {
    links_a += a.neighbors(id, 0).size();
    links_b += b.neighbors(id, 0).size();
  }
  CHECK(links_b > links_a);
}

TEST_CASE("reordering preserves results and groups neighbours closer in id space") {
  const auto data = clustered(kCount, kDim, 11);
  const auto queries = clustered(50, kDim, 12);
  Index index = build(data, Metric::L2);

  const auto mean_gap = [&](const Index& ix) {
    double total = 0.0;
    std::size_t edges = 0;
    for (std::uint32_t id = 0; id < ix.size(); ++id) {
      for (const std::uint32_t neighbor : ix.neighbors(id, 0)) {
        total += std::abs(static_cast<double>(id) - static_cast<double>(neighbor));
        ++edges;
      }
    }
    return total / static_cast<double>(edges);
  };
  const double gap_before = mean_gap(index);

  std::vector<std::vector<Neighbor>> before(50, std::vector<Neighbor>(10));
  for (std::size_t q = 0; q < 50; ++q) {
    index.search(std::span<const float>(queries).subspan(q * kDim, kDim), 64, before[q]);
  }
  index.reorder();
  CHECK(index.verify().ok);
  CHECK(mean_gap(index) < gap_before);
  for (std::size_t q = 0; q < 50; ++q) {
    std::vector<Neighbor> after(10);
    index.search(std::span<const float>(queries).subspan(q * kDim, kDim), 64, after);
    for (std::size_t i = 0; i < 10; ++i) {
      CHECK(after[i].distance == doctest::Approx(before[q][i].distance));
    }
  }
}

TEST_CASE("an index keeps accepting points after a reorder") {
  const auto data = clustered(600, kDim, 13);
  Index index(options_for(Metric::L2, 600));
  index.add_batch(std::span<const float>(data).subspan(0, 400 * kDim), 1);
  index.reorder();
  for (std::size_t i = 400; i < 600; ++i) {
    index.add(std::span<const float>(data).subspan(i * kDim, kDim), i);
  }
  CHECK(index.size() == 600);
  CHECK(index.verify().ok);
  std::vector<Neighbor> out(1);
  REQUIRE(index.search(std::span<const float>(data).subspan(599 * kDim, kDim), 100, out) == 1);
  CHECK(out[0].label == 599);
}

TEST_CASE("prefetch depth changes nothing but speed") {
  const auto data = clustered(1000, kDim, 14);
  Index index = build(data, Metric::L2);
  const auto query = std::span<const float>(data).subspan(0, kDim);
  std::vector<Neighbor> baseline(10);
  index.search(query, 64, baseline);
  for (const std::size_t lines : {0, 1, 8}) {
    index.set_prefetch_lines(lines);
    std::vector<Neighbor> out(10);
    index.search(query, 64, out);
    for (std::size_t i = 0; i < 10; ++i) {
      CHECK(out[i].label == baseline[i].label);
    }
  }
}
