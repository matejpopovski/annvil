#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/pq.hpp"
#include "detail/format.hpp"
#include "detail/region.hpp"

namespace annvil {

inline constexpr std::uint32_t kMaxM = 128;
inline constexpr std::uint32_t kMaxM0 = 2 * kMaxM;
inline constexpr std::size_t kMaxDim = 65536;
inline constexpr int kMaxLevel = 63;
inline constexpr std::uint64_t kNoEntry = ~std::uint64_t{0};
inline constexpr std::size_t kMaxCapacity = (std::size_t{1} << 31) - 1;

#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::size_t kCacheLine = 128;
#else
inline constexpr std::size_t kCacheLine = 64;
#endif
// Bounds the work per neighbour; past this size the hardware prefetcher has the
// stream anyway.
inline constexpr std::size_t kMaxPrefetchLines = 16;

inline constexpr std::uint64_t pack_entry(std::uint32_t id, int level) noexcept {
  return (static_cast<std::uint64_t>(level) << 32) | id;
}
inline constexpr std::uint32_t entry_id(std::uint64_t packed) noexcept {
  return static_cast<std::uint32_t>(packed);
}
inline constexpr int entry_level(std::uint64_t packed) noexcept {
  return static_cast<int>(packed >> 32);
}

// Reader/writer protocol for link slots. Every slot is written with release after
// the node it names is complete, and a list's count is written last, so an acquire
// load of an id synchronizes with the vector, label and list behind it. Slots are
// never cleared and nodes are never removed: a reader holding a stale count sees
// old but valid ids, and the visited set absorbs duplicates. On aarch64 this is an
// LDAPR per slot; on x86 it is a plain MOV.
inline std::uint32_t load_slot(const std::uint32_t& slot) noexcept {
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(slot))
      .load(std::memory_order_acquire);
}
inline void store_slot(std::uint32_t& slot, std::uint32_t value) noexcept {
  std::atomic_ref<std::uint32_t>(slot).store(value, std::memory_order_release);
}

struct alignas(64) PaddedMutex {
  std::mutex mutex;
};

struct Index::Impl {
  // Fixed configuration.
  std::size_t dim = 0;
  std::size_t kernel_dim = 0;  // dim rounded to whole SIMD lanes; the padding is zero.
  std::size_t stride = 0;      // Floats per stored vector.
  std::size_t prefetch_lines = 0;
  Metric metric = Metric::L2;
  std::uint32_t M = 0;
  std::uint32_t M0 = 0;
  std::uint32_t ef_construction = 0;
  std::uint64_t seed = 0;
  double level_multiplier = 0.0;
  bool keep_pruned = false;
  std::size_t capacity = 0;
  std::size_t link0_slots = 0;  // 1 + M0
  std::size_t upper_slots = 0;  // 1 + M
  std::size_t total_upper_slots = 0;
  bool read_only = false;

  // Flat arrays, pointing into `region`.
  detail::Region region;
  std::uint8_t* levels = nullptr;
  std::uint32_t* upper_offsets = nullptr;
  std::uint32_t* links0 = nullptr;
  std::uint32_t* upper = nullptr;
  std::uint64_t* labels = nullptr;
  float* vectors = nullptr;

  // Quantizer state. Codes live in `owned_codes` until saved, then in the mapping.
  bool has_pq = false;
  ProductQuantizer pq;
  std::vector<std::uint8_t> owned_codes;
  const std::uint8_t* codes = nullptr;

  // Set for mapped indexes, zero otherwise.
  std::uint64_t payload_checksum = 0;

  std::atomic<std::uint64_t> entry{kNoEntry};
  std::atomic<std::uint32_t> next_id{0};
  std::atomic<std::uint32_t> published{0};

  // Striped because only one node lock is ever held at a time, so collisions cost
  // a little waiting but cannot deadlock.
  static constexpr std::size_t kLockStripes = 4096;
  std::unique_ptr<PaddedMutex[]> locks;

  const float* vector(std::uint32_t id) const noexcept { return vectors + id * stride; }

  const std::uint32_t* list(std::uint32_t id, int level) const noexcept {
    return level == 0
               ? links0 + id * link0_slots
               : upper + upper_offsets[id] + static_cast<std::size_t>(level - 1) * upper_slots;
  }
  std::uint32_t* list(std::uint32_t id, int level) noexcept {
    return const_cast<std::uint32_t*>(static_cast<const Impl*>(this)->list(id, level));
  }
  std::uint32_t max_links(int level) const noexcept { return level == 0 ? M0 : M; }

  void set_stride(std::size_t floats) noexcept {
    stride = floats;
    const std::size_t bytes = floats * sizeof(float);
    prefetch_lines = std::min((bytes + kCacheLine - 1) / kCacheLine, kMaxPrefetchLines);
  }

  void bind(detail::Region&& mapped, const detail::Layout& layout) noexcept {
    region = std::move(mapped);
    std::byte* base = region.data();
    levels = reinterpret_cast<std::uint8_t*>(base + layout.levels.offset);
    upper_offsets = reinterpret_cast<std::uint32_t*>(base + layout.upper_offsets.offset);
    links0 = reinterpret_cast<std::uint32_t*>(base + layout.links0.offset);
    upper = reinterpret_cast<std::uint32_t*>(base + layout.upper.offset);
    labels = reinterpret_cast<std::uint64_t*>(base + layout.labels.offset);
    vectors = reinterpret_cast<float*>(base + layout.vectors.offset);
  }
};

}  // namespace annvil
