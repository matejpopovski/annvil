#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/pq.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 32;

double mean_reconstruction_error(const ProductQuantizer& pq, const std::vector<float>& data) {
  std::vector<std::uint8_t> code(pq.m());
  std::vector<float> decoded(kDim);
  double total = 0.0;
  const std::size_t n = data.size() / kDim;
  for (std::size_t i = 0; i < n; ++i) {
    pq.encode(&data[i * kDim], code.data());
    pq.decode(code.data(), decoded.data());
    total += kernels::l2_sq_scalar(&data[i * kDim], decoded.data(), kDim);
  }
  return total / static_cast<double>(n);
}

PqTrainOptions train_options(std::size_t m) {
  PqTrainOptions o;
  o.m = m;
  o.iterations = 10;
  o.threads = 2;
  return o;
}

}  // namespace

TEST_CASE("more subspaces reconstruct vectors more accurately") {
  const auto data = clustered(3000, kDim, 21);
  const auto coarse = ProductQuantizer::train(data.data(), 3000, kDim, kDim, train_options(4));
  const auto fine = ProductQuantizer::train(data.data(), 3000, kDim, kDim, train_options(16));
  CHECK(mean_reconstruction_error(fine, data) < mean_reconstruction_error(coarse, data));
}

TEST_CASE("the L2 table sums to the distance from the query to the decoded vector") {
  const auto data = clustered(1000, kDim, 22);
  const auto pq = ProductQuantizer::train(data.data(), 1000, kDim, kDim, train_options(8));
  std::vector<float> table(pq.table_size());
  pq.l2_table(&data[0], table.data());
  std::vector<std::uint8_t> code(pq.m());
  std::vector<float> decoded(kDim);
  pq.encode(&data[kDim], code.data());
  pq.decode(code.data(), decoded.data());
  float sum = 0.0f;
  for (std::size_t j = 0; j < pq.m(); ++j) {
    sum += table[j * kPqCentroids + code[j]];
  }
  CHECK(sum ==
        doctest::Approx(kernels::l2_sq_scalar(&data[0], decoded.data(), kDim)).epsilon(1e-3));
}

TEST_CASE("the dot table sums to the dot product with the decoded vector") {
  auto data = clustered(1000, kDim, 23);
  normalize_rows(data, kDim);
  const auto pq = ProductQuantizer::train(data.data(), 1000, kDim, kDim, train_options(8));
  std::vector<float> table(pq.table_size());
  pq.dot_table(&data[0], table.data());
  std::vector<std::uint8_t> code(pq.m());
  std::vector<float> decoded(kDim);
  pq.encode(&data[kDim], code.data());
  pq.decode(code.data(), decoded.data());
  float sum = 0.0f;
  for (std::size_t j = 0; j < pq.m(); ++j) {
    sum += table[j * kPqCentroids + code[j]];
  }
  CHECK(sum == doctest::Approx(kernels::dot_scalar(&data[0], decoded.data(), kDim)).epsilon(1e-3));
}

TEST_CASE("subspaces cover every dimension when m does not divide dim") {
  const std::size_t dim = 25;
  const auto data = clustered(1000, dim, 24);
  PqTrainOptions o = train_options(8);
  const auto pq = ProductQuantizer::train(data.data(), 1000, dim, dim, o);
  CHECK(pq.codebook().size() == 256 * dim);
  std::vector<std::uint8_t> code(8);
  std::vector<float> decoded(dim, std::numeric_limits<float>::quiet_NaN());
  pq.encode(&data[0], code.data());
  pq.decode(code.data(), decoded.data());
  for (const float x : decoded) {
    CHECK(std::isfinite(x));
  }
}

TEST_CASE("training is reproducible for a fixed seed") {
  const auto data = clustered(1000, kDim, 25);
  const auto a = ProductQuantizer::train(data.data(), 1000, kDim, kDim, train_options(8));
  const auto b = ProductQuantizer::train(data.data(), 1000, kDim, kDim, train_options(8));
  CHECK(std::equal(a.codebook().begin(), a.codebook().end(), b.codebook().begin()));
}

TEST_CASE("training rejects unusable parameters") {
  const auto data = clustered(100, kDim, 26);
  CHECK_THROWS_AS(ProductQuantizer::train(data.data(), 100, kDim, kDim, train_options(8)),
                  std::invalid_argument);
  const auto enough = clustered(300, kDim, 26);
  CHECK_THROWS_AS(ProductQuantizer::train(enough.data(), 300, kDim, kDim, train_options(0)),
                  std::invalid_argument);
  CHECK_THROWS_AS(ProductQuantizer::train(enough.data(), 300, kDim, kDim, train_options(kDim + 1)),
                  std::invalid_argument);
}

TEST_CASE("graph traversal on PQ codes with reranking recovers exact recall") {
  for (const Metric metric : {Metric::L2, Metric::Angular}) {
    CAPTURE(static_cast<int>(metric));
    auto data = clustered(4000, kDim, 27);
    auto queries = clustered(100, kDim, 28);
    if (metric == Metric::Angular) {
      normalize_rows(data, kDim);
      normalize_rows(queries, kDim);
    }
    IndexOptions o;
    o.dim = kDim;
    o.metric = metric;
    o.capacity = 4000;
    o.M = 12;
    o.ef_construction = 100;
    Index index(o);
    index.add_batch(data, 2);
    index.attach_pq(train_options(16));
    CHECK(index.has_pq());
    CHECK(index.memory().pq == 4000 * 16 + 256 * kDim * sizeof(float));

    const double reranked = mean_recall(data, queries, kDim, 10, metric,
                                        [&](std::span<const float> q, std::span<Neighbor> out) {
                                          return index.search_pq(q, 200, 100, out);
                                        });
    const double unranked = mean_recall(data, queries, kDim, 10, metric,
                                        [&](std::span<const float> q, std::span<Neighbor> out) {
                                          return index.search_pq(q, 200, 0, out);
                                        });
    CHECK(reranked > 0.95);
    CHECK(reranked >= unranked);
  }
}

TEST_CASE("PQ search needs a trained quantizer and enough points") {
  IndexOptions o;
  o.dim = kDim;
  o.capacity = 10;
  Index index(o);
  const std::vector<float> row(kDim, 1.0f);
  index.add(row, 0);
  std::vector<Neighbor> out(1);
  CHECK_THROWS_AS(index.search_pq(row, 10, 5, out), std::logic_error);
  CHECK_THROWS_AS(index.attach_pq(train_options(8)), std::invalid_argument);
}
