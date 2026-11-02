// The interface the harness drives. Each engine is built, saved and loaded through it,
// so all timing code lives in one place and applies to every engine alike.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "annvil/index.hpp"

namespace annvil::bench {

struct EngineConfig {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t capacity = 0;
  std::uint32_t M = 16;
  std::uint32_t ef_construction = 200;
};

class Engine {
 public:
  virtual ~Engine() = default;

  // `vectors` holds `n` rows and, for the angular metric, is already normalized.
  virtual void build(const float* vectors, std::size_t n, std::size_t threads) = 0;
  virtual void save(const std::filesystem::path& path) = 0;
  virtual void load(const std::filesystem::path& path) = 0;

  virtual void set_ef(std::size_t ef) = 0;
  // Writes up to k labels, closest first. Thread-safe once built.
  virtual std::size_t search(const float* query, std::size_t k, std::uint64_t* labels) const = 0;
  // Same search with distance computations counted, for the untimed quality pass.
  virtual std::size_t search_counted(const float* query, std::size_t k, std::uint64_t* labels,
                                     std::uint64_t* distance_computations) const = 0;

  // Bytes the engine's own data structures hold, by its layout rather than by RSS.
  virtual std::size_t memory_bytes() const = 0;
};

// The distance function an engine actually dispatches to, for the kernel microbenchmark.
using RawDistance = float (*)(const void* a, const void* b, const void* param);
struct RawKernel {
  RawDistance function;
  const void* param;
};

std::unique_ptr<Engine> make_hnswlib_engine(const EngineConfig& config);
std::unique_ptr<Engine> make_hnswlib_tuned_engine(const EngineConfig& config);
RawKernel hnswlib_raw_kernel(std::size_t dim, Metric metric);

}  // namespace annvil::bench
