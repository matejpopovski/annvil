// Compiled twice: once against upstream hnswlib and once against the tuned copy
// (see bench/CMakeLists.txt). Everything else about the two engines is identical.
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

#include "annvil/kernels.hpp"
#include "annvil/parallel.hpp"
#include "engine.hpp"

namespace annvil::bench {
namespace {

#if defined(ANNVIL_HNSWLIB_TUNED)
float tuned_l2(const void* a, const void* b, const void* dim) {
  return kernels::l2_sq(static_cast<const float*>(a), static_cast<const float*>(b),
                        *static_cast<const std::size_t*>(dim));
}
float tuned_ip(const void* a, const void* b, const void* dim) {
  return 1.0f - kernels::dot(static_cast<const float*>(a), static_cast<const float*>(b),
                             *static_cast<const std::size_t*>(dim));
}

class KernelSpace final : public hnswlib::SpaceInterface<float> {
 public:
  KernelSpace(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {}
  std::size_t get_data_size() override { return dim_ * sizeof(float); }
  hnswlib::DISTFUNC<float> get_dist_func() override {
    return metric_ == Metric::L2 ? tuned_l2 : tuned_ip;
  }
  void* get_dist_func_param() override { return &dim_; }

 private:
  std::size_t dim_;
  Metric metric_;
};
#endif

std::unique_ptr<hnswlib::SpaceInterface<float>> make_space(const EngineConfig& config) {
#if defined(ANNVIL_HNSWLIB_TUNED)
  return std::make_unique<KernelSpace>(config.dim, config.metric);
#else
  if (config.metric == Metric::L2) {
    return std::make_unique<hnswlib::L2Space>(config.dim);
  }
  return std::make_unique<hnswlib::InnerProductSpace>(config.dim);
#endif
}

// Wraps the distance function during the counted pass only, leaving the timed path
// exactly as shipped.
thread_local std::uint64_t t_distance_count = 0;
thread_local hnswlib::DISTFUNC<float> t_inner = nullptr;

float counting_distance(const void* a, const void* b, const void* param) {
  ++t_distance_count;
  return t_inner(a, b, param);
}

class HnswlibEngine final : public Engine {
 public:
  explicit HnswlibEngine(const EngineConfig& config)
      : config_(config), space_(make_space(config)) {}

  void build(const float* vectors, std::size_t n, std::size_t threads) override {
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), config_.capacity,
                                                               config_.M, config_.ef_construction);
    parallel_for(n, threads, [&](std::size_t i, std::size_t) {
      index_->addPoint(vectors + i * config_.dim, static_cast<hnswlib::labeltype>(i));
    });
  }

  void save(const std::filesystem::path& path) override { index_->saveIndex(path.string()); }

  void load(const std::filesystem::path& path) override {
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), path.string());
  }

  void set_ef(std::size_t ef) override { index_->setEf(ef); }

  std::size_t search(const float* query, std::size_t k, std::uint64_t* labels) const override {
    auto heap = index_->searchKnn(query, k);
    std::size_t found = heap.size();
    for (std::size_t i = found; i > 0; --i) {
      labels[i - 1] = heap.top().second;
      heap.pop();
    }
    return found;
  }

  std::size_t search_counted(const float* query, std::size_t k, std::uint64_t* labels,
                             std::uint64_t* distance_computations) const override {
    const hnswlib::DISTFUNC<float> original = index_->fstdistfunc_;
    t_inner = original;
    t_distance_count = 0;
    index_->fstdistfunc_ = counting_distance;
    const std::size_t found = search(query, k, labels);
    index_->fstdistfunc_ = original;
    *distance_computations += t_distance_count;
    return found;
  }

  // Follows hnswlib's own allocations: one flat level-0 block, one malloc per node with
  // upper levels, plus a per-node mutex, the label map and two lock tables.
  std::size_t memory_bytes() const override {
    const std::size_t n = index_->cur_element_count;
    std::size_t bytes = n * index_->size_data_per_element_;
    for (std::size_t i = 0; i < n; ++i) {
      bytes += sizeof(void*);
      if (index_->element_levels_[i] > 0) {
        bytes +=
            static_cast<std::size_t>(index_->element_levels_[i]) * index_->size_links_per_element_;
      }
      bytes += sizeof(int);
    }
    bytes += n * sizeof(std::mutex);
    bytes += index_->label_op_locks_.size() * sizeof(std::mutex);
    bytes += n * (sizeof(hnswlib::labeltype) + sizeof(hnswlib::tableint) + 2 * sizeof(void*));
    return bytes;
  }

 private:
  EngineConfig config_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

}  // namespace

std::unique_ptr<Engine> ANNVIL_ENGINE_FACTORY(const EngineConfig& config) {
  return std::make_unique<HnswlibEngine>(config);
}

#if !defined(ANNVIL_HNSWLIB_TUNED)
RawKernel hnswlib_raw_kernel(std::size_t dim, Metric metric) {
  // Leaked on purpose: the caller keeps the parameter pointer for the process lifetime.
  EngineConfig config;
  config.dim = dim;
  config.metric = metric;
  auto* space = make_space(config).release();
  return {space->get_dist_func(), space->get_dist_func_param()};
}
#endif

}  // namespace annvil::bench
