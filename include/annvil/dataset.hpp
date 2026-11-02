#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "annvil/index.hpp"

namespace annvil {

// Read-only view of a .annb file: one ann-benchmarks HDF5 dataset flattened into
// raw little-endian arrays by scripts/hdf5_to_annb.py, so the harness needs no HDF5
// dependency.
//
//   offset 0   8 bytes  magic "ANNB0001"
//   offset 8   u32      metric (0 = euclidean, 1 = angular)
//   offset 12  u32      dim
//   offset 16  u64      train count
//   offset 24  u64      test count
//   offset 32  u64      ground-truth neighbours per query (gt_k)
//   offset 40  24 bytes zero padding
//   offset 64  f32[train * dim], f32[test * dim], i32[test * gt_k], f32[test * gt_k]
class Dataset {
 public:
  static Dataset open(const std::filesystem::path& path);
  Dataset(Dataset&&) noexcept;
  Dataset& operator=(Dataset&&) noexcept;
  ~Dataset();

  Metric metric() const noexcept { return metric_; }
  std::size_t dim() const noexcept { return dim_; }
  std::size_t train_count() const noexcept { return train_count_; }
  std::size_t test_count() const noexcept { return test_count_; }
  std::size_t gt_k() const noexcept { return gt_k_; }

  const float* train() const noexcept { return train_; }
  const float* test() const noexcept { return test_; }
  const float* test_vector(std::size_t i) const noexcept { return test_ + i * dim_; }
  const float* train_vector(std::size_t i) const noexcept { return train_ + i * dim_; }
  const std::int32_t* gt_neighbors(std::size_t i) const noexcept { return gt_ids_ + i * gt_k_; }
  const float* gt_distances(std::size_t i) const noexcept { return gt_dists_ + i * gt_k_; }

 private:
  Dataset() = default;

  struct Mapping;
  std::unique_ptr<Mapping> mapping_;
  Metric metric_ = Metric::L2;
  std::size_t dim_ = 0;
  std::size_t train_count_ = 0;
  std::size_t test_count_ = 0;
  std::size_t gt_k_ = 0;
  const float* train_ = nullptr;
  const float* test_ = nullptr;
  const std::int32_t* gt_ids_ = nullptr;
  const float* gt_dists_ = nullptr;
};

}  // namespace annvil
