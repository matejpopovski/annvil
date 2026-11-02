#pragma once

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/kernels.hpp"

namespace annvil::test {

// Clustered Gaussian data. Uniform random vectors in high dimension make every point
// nearly equidistant, which is the one case where HNSW has nothing to exploit.
inline std::vector<float> clustered(std::size_t n, std::size_t dim, std::uint64_t seed,
                                    std::size_t clusters = 32) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> centre(0.0f, 3.0f);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  std::vector<float> centres(clusters * dim);
  for (float& x : centres) {
    x = centre(rng);
  }
  std::vector<float> data(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = rng() % clusters;
    for (std::size_t d = 0; d < dim; ++d) {
      data[i * dim + d] = centres[c * dim + d] + noise(rng);
    }
  }
  return data;
}

inline void normalize_rows(std::vector<float>& data, std::size_t dim) {
  for (std::size_t i = 0; i * dim < data.size(); ++i) {
    float* row = &data[i * dim];
    const float norm = std::sqrt(kernels::dot_scalar(row, row, dim));
    for (std::size_t d = 0; d < dim && norm > 0.0f; ++d) {
      row[d] /= norm;
    }
  }
}

// Exact k nearest labels by linear scan, the reference every recall figure is against.
inline std::vector<std::uint64_t> exact_knn(std::span<const float> data, std::size_t dim,
                                            const float* query, std::size_t k, Metric metric) {
  const std::size_t n = data.size() / dim;
  std::vector<std::pair<float, std::uint64_t>> scored(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = &data[i * dim];
    const float d = metric == Metric::L2 ? kernels::l2_sq_scalar(query, row, dim)
                                         : 1.0f - kernels::dot_scalar(query, row, dim);
    scored[i] = {d, i};
  }
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end());
  std::vector<std::uint64_t> labels(k);
  for (std::size_t i = 0; i < k; ++i) {
    labels[i] = scored[i].second;
  }
  return labels;
}

template <class Search>
double mean_recall(std::span<const float> data, std::span<const float> queries, std::size_t dim,
                   std::size_t k, Metric metric, Search&& search) {
  const std::size_t q = queries.size() / dim;
  std::size_t hits = 0;
  std::vector<Neighbor> out(k);
  for (std::size_t i = 0; i < q; ++i) {
    const auto truth = exact_knn(data, dim, &queries[i * dim], k, metric);
    const std::set<std::uint64_t> wanted(truth.begin(), truth.end());
    const std::size_t found = search(queries.subspan(i * dim, dim), std::span<Neighbor>(out));
    for (std::size_t j = 0; j < found; ++j) {
      hits += wanted.count(out[j].label);
    }
  }
  return static_cast<double>(hits) / static_cast<double>(q * k);
}

// How many nodes no search could ever return: not reachable from the entry point
// through any layer's links.
inline std::size_t unreachable_nodes(const Index& index) {
  std::vector<char> seen(index.size(), 0);
  std::vector<std::uint32_t> stack = {index.entry_node()};
  seen[stack.back()] = 1;
  std::size_t reached = 1;
  while (!stack.empty()) {
    const std::uint32_t node = stack.back();
    stack.pop_back();
    for (int level = index.node_level(node); level >= 0; --level) {
      for (const std::uint32_t next : index.neighbors(node, level)) {
        if (seen[next] == 0) {
          seen[next] = 1;
          ++reached;
          stack.push_back(next);
        }
      }
    }
  }
  return index.size() - reached;
}

class TempDir {
 public:
  TempDir() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("annvil-test-" + std::to_string(::getpid()) + "-" + std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::filesystem::path file(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

}  // namespace annvil::test
