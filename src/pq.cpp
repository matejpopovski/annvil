#include "annvil/pq.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "annvil/kernels.hpp"
#include "annvil/parallel.hpp"

namespace annvil {
namespace {

// Lloyd's algorithm on one subspace. `points` holds `count` rows of `sub_dim` floats.
void kmeans(const std::vector<float>& points, std::size_t count, std::size_t sub_dim,
            std::size_t iterations, std::uint64_t seed, float* centroids) {
  std::mt19937_64 rng(seed);
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::shuffle(order.begin(), order.end(), rng);
  for (std::size_t c = 0; c < kPqCentroids; ++c) {
    std::copy_n(&points[order[c] * sub_dim], sub_dim, centroids + c * sub_dim);
  }

  std::vector<std::uint32_t> assignment(count);
  std::vector<double> sums(kPqCentroids * sub_dim);
  std::vector<std::size_t> sizes(kPqCentroids);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(sizes.begin(), sizes.end(), std::size_t{0});
    bool changed = iteration == 0;
    for (std::size_t i = 0; i < count; ++i) {
      const float* point = &points[i * sub_dim];
      float best = std::numeric_limits<float>::infinity();
      std::uint32_t best_c = 0;
      for (std::uint32_t c = 0; c < kPqCentroids; ++c) {
        const float d = kernels::l2_sq(point, centroids + c * sub_dim, sub_dim);
        if (d < best) {
          best = d;
          best_c = c;
        }
      }
      changed = changed || assignment[i] != best_c;
      assignment[i] = best_c;
      ++sizes[best_c];
      for (std::size_t d = 0; d < sub_dim; ++d) {
        sums[best_c * sub_dim + d] += point[d];
      }
    }
    if (!changed) {
      break;
    }
    for (std::size_t c = 0; c < kPqCentroids; ++c) {
      if (sizes[c] != 0) {
        for (std::size_t d = 0; d < sub_dim; ++d) {
          centroids[c * sub_dim + d] =
              static_cast<float>(sums[c * sub_dim + d] / static_cast<double>(sizes[c]));
        }
      } else {
        // An empty centroid wastes a code. Re-seeding it from a random point of the
        // largest cluster moves that capacity to where the data is densest.
        const auto largest =
            static_cast<std::size_t>(std::max_element(sizes.begin(), sizes.end()) - sizes.begin());
        std::vector<std::size_t> members;
        for (std::size_t i = 0; i < count; ++i) {
          if (assignment[i] == largest) {
            members.push_back(i);
          }
        }
        const std::size_t pick = members[rng() % members.size()];
        std::copy_n(&points[pick * sub_dim], sub_dim, centroids + c * sub_dim);
      }
    }
  }
}

}  // namespace

ProductQuantizer::ProductQuantizer(std::size_t dim, std::size_t m) : dim_(dim), m_(m) {
  if (m == 0 || m > dim) {
    throw std::invalid_argument("pq m must be in [1, dim]");
  }
  sub_offset_.resize(m);
  sub_dim_.resize(m);
  codebook_offset_.resize(m);
  const std::size_t base = dim / m;
  const std::size_t extra = dim % m;
  std::size_t offset = 0;
  for (std::size_t j = 0; j < m; ++j) {
    sub_offset_[j] = offset;
    sub_dim_[j] = base + (j < extra ? 1 : 0);
    codebook_offset_[j] = offset * kPqCentroids;
    offset += sub_dim_[j];
  }
  codebook_.resize(dim * kPqCentroids);
}

ProductQuantizer ProductQuantizer::train(const float* data, std::size_t n, std::size_t stride,
                                         std::size_t dim, const PqTrainOptions& options) {
  if (n < kPqCentroids) {
    throw std::invalid_argument("pq training needs at least 256 vectors");
  }
  ProductQuantizer pq(dim, options.m);

  std::mt19937_64 rng(options.seed);
  std::vector<std::size_t> sample(n);
  std::iota(sample.begin(), sample.end(), std::size_t{0});
  const std::size_t count = std::clamp(options.max_training_points, kPqCentroids, n);
  for (std::size_t i = 0; i < count; ++i) {
    std::swap(sample[i], sample[i + rng() % (n - i)]);
  }

  parallel_for(
      options.m, options.threads,
      [&](std::size_t j, std::size_t) {
        const std::size_t sub_dim = pq.sub_dim_[j];
        std::vector<float> points(count * sub_dim);
        for (std::size_t i = 0; i < count; ++i) {
          std::copy_n(data + sample[i] * stride + pq.sub_offset_[j], sub_dim, &points[i * sub_dim]);
        }
        kmeans(points, count, sub_dim, options.iterations, options.seed + j + 1,
               pq.codebook_.data() + pq.codebook_offset_[j]);
      },
      1);
  return pq;
}

ProductQuantizer ProductQuantizer::from_codebook(std::size_t dim, std::size_t m,
                                                 std::vector<float> codebook) {
  ProductQuantizer pq(dim, m);
  if (codebook.size() != pq.codebook_.size()) {
    throw std::invalid_argument("codebook size does not match dim");
  }
  pq.codebook_ = std::move(codebook);
  return pq;
}

const float* ProductQuantizer::centroid(std::size_t subspace, std::size_t index) const noexcept {
  return codebook_.data() + codebook_offset_[subspace] + index * sub_dim_[subspace];
}

void ProductQuantizer::encode(const float* vector, std::uint8_t* code) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) {
    const float* sub = vector + sub_offset_[j];
    float best = std::numeric_limits<float>::infinity();
    std::size_t best_c = 0;
    for (std::size_t c = 0; c < kPqCentroids; ++c) {
      const float d = kernels::l2_sq(sub, centroid(j, c), sub_dim_[j]);
      if (d < best) {
        best = d;
        best_c = c;
      }
    }
    code[j] = static_cast<std::uint8_t>(best_c);
  }
}

void ProductQuantizer::decode(const std::uint8_t* code, float* vector) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) {
    std::copy_n(centroid(j, code[j]), sub_dim_[j], vector + sub_offset_[j]);
  }
}

void ProductQuantizer::l2_table(const float* query, float* table) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) {
    for (std::size_t c = 0; c < kPqCentroids; ++c) {
      table[j * kPqCentroids + c] =
          kernels::l2_sq(query + sub_offset_[j], centroid(j, c), sub_dim_[j]);
    }
  }
}

void ProductQuantizer::dot_table(const float* query, float* table) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) {
    for (std::size_t c = 0; c < kPqCentroids; ++c) {
      table[j * kPqCentroids + c] =
          kernels::dot(query + sub_offset_[j], centroid(j, c), sub_dim_[j]);
    }
  }
}

}  // namespace annvil
