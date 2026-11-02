// On-disk format. The file is the in-memory layout with a header in front, so these
// structs are the only serialization code in the project.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace annvil::detail {

inline constexpr char kMagic[8] = {'A', 'N', 'N', 'V', 'I', 'L', 0x1a, '\n'};
inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint32_t kEndianTag = 0x01020304;
inline constexpr std::size_t kSectionAlign = 4096;

inline constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

struct Section {
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
};

struct Layout {
  Section levels;         // u8[n]
  Section upper_offsets;  // u32[n], slot index of each node's first upper list
  Section links0;         // u32[n * (1 + M0)], [count, neighbour...] per node
  Section upper;          // u32[upper_slots], [count, neighbour...] per level per node
  Section labels;         // u64[n]
  Section vectors;        // f32[n * stride]
  Section pq_codebook;    // f32[256 * dim]
  Section pq_codes;       // u8[n * pq_m]
  std::uint64_t total = 0;
};

struct LayoutInput {
  std::uint64_t nodes = 0;
  std::uint64_t link0_slots = 0;  // 1 + M0
  std::uint64_t upper_slots = 0;
  std::uint64_t stride = 0;  // Floats per stored vector.
  std::uint64_t dim = 0;
  std::uint64_t pq_m = 0;
};

inline Layout compute_layout(const LayoutInput& in) noexcept {
  Layout out;
  std::uint64_t cursor = kSectionAlign;  // The header occupies the first block.
  auto place = [&cursor](Section& section, std::uint64_t size) {
    section.offset = cursor;
    section.size = size;
    cursor = align_up(cursor + size, kSectionAlign);
  };
  place(out.levels, in.nodes);
  place(out.upper_offsets, in.nodes * sizeof(std::uint32_t));
  place(out.links0, in.nodes * in.link0_slots * sizeof(std::uint32_t));
  place(out.upper, in.upper_slots * sizeof(std::uint32_t));
  place(out.labels, in.nodes * sizeof(std::uint64_t));
  place(out.vectors, in.nodes * in.stride * sizeof(float));
  place(out.pq_codebook, in.pq_m != 0 ? 256 * in.dim * sizeof(float) : 0);
  place(out.pq_codes, in.pq_m != 0 ? in.nodes * in.pq_m : 0);
  out.total = cursor;
  return out;
}

struct FileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t endian_tag;
  std::uint32_t dim;
  std::uint32_t stride;
  std::uint32_t metric;
  std::uint32_t M;
  std::uint32_t M0;
  std::uint32_t ef_construction;
  std::uint32_t pq_m;
  std::uint32_t reserved;
  std::uint64_t count;
  std::uint64_t seed;
  std::uint64_t entry;
  std::uint64_t upper_slots;
  double level_multiplier;
  std::uint64_t file_size;
  std::uint64_t payload_checksum;
  std::uint64_t header_checksum;  // Over every byte before this field.
};
static_assert(sizeof(FileHeader) == 112);

// FNV-1a widened to 64-bit lanes. Multiplying by an odd constant is a bijection, so
// any change to a single lane always changes the result; this guards against disk
// corruption and truncation, not against an adversary.
inline std::uint64_t hash_bytes(const void* data, std::size_t size,
                                std::uint64_t seed = 0xcbf29ce484222325ULL) noexcept {
  constexpr std::uint64_t kPrime = 0x100000001b3ULL;
  const auto* bytes = static_cast<const unsigned char*>(data);
  std::uint64_t hash = seed;
  std::size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    std::uint64_t lane;
    std::memcpy(&lane, bytes + i, 8);
    hash = (hash ^ lane) * kPrime;
  }
  for (; i < size; ++i) {
    hash = (hash ^ bytes[i]) * kPrime;
  }
  return hash;
}

}  // namespace annvil::detail
