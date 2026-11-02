// Per-thread search state. Both structures replace something hnswlib keeps in larger
// or pointer-heavier form; the benchmark's "hnswlib-tuned" engine shows what that is
// worth with the distance kernels held equal.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace annvil::detail {

// One bit per node: about 150 KB at 1.2M nodes, against 2.4 MB for a 16-bit tag per
// node. Clearing walks only the words that were touched, so a query's cost tracks
// its own visits rather than the index size.
class VisitedSet {
 public:
  void prepare(std::size_t capacity) {
    const std::size_t words = (capacity + 63) / 64;
    if (words_.size() < words) {
      words_.assign(words, 0);
      dirty_.assign(words, 0);
      dirty_count_ = 0;
    }
    for (std::size_t i = 0; i < dirty_count_; ++i) {
      words_[dirty_[i]] = 0;
    }
    dirty_count_ = 0;
  }

  // Returns true if `id` had not been seen yet.
  bool insert(std::uint32_t id) noexcept {
    std::uint64_t& word = words_[id >> 6];
    const std::uint64_t mask = std::uint64_t{1} << (id & 63);
    if ((word & mask) != 0) {
      return false;
    }
    if (word == 0) {
      dirty_[dirty_count_++] = id >> 6;
    }
    word |= mask;
    return true;
  }

 private:
  std::vector<std::uint64_t> words_;
  std::vector<std::uint32_t> dirty_;
  std::size_t dirty_count_ = 0;
};

struct BeamEntry {
  float distance;
  std::uint32_t tagged_id;  // Low 31 bits are the node, the top bit means "expanded".
};

// The ef best nodes seen so far, sorted by distance, with a cursor at the closest
// node not yet expanded. This single array does the job of hnswlib's two heaps,
// because a candidate that falls outside the top ef can never be expanded anyway
// (Algorithm 2's stop condition). Inserting is a binary search and a memmove of
// 8-byte entries.
class Beam {
 public:
  static constexpr std::uint32_t kExpanded = 0x80000000u;

  void reset(std::size_t ef) {
    if (entries_.size() < ef) {
      entries_.resize(ef);
    }
    ef_ = ef;
    size_ = 0;
    cursor_ = 0;
  }

  // Candidates at or beyond this distance would be evicted immediately. The sentinel
  // is max() rather than infinity so that a host project building this file with
  // -ffast-math, which assumes infinities do not occur, still gets working code.
  float bound() const noexcept {
    return size_ < ef_ ? std::numeric_limits<float>::max() : entries_[size_ - 1].distance;
  }

  void insert(float distance, std::uint32_t id) noexcept {
    std::size_t lo = 0;
    std::size_t len = size_;
    while (len > 0) {
      const std::size_t half = len / 2;
      const std::size_t step = entries_[lo + half].distance < distance ? half + (len & 1) : 0;
      lo += step;
      len = half;
    }
    const std::size_t new_size = std::min(size_ + 1, ef_);
    if (lo >= new_size) {
      return;
    }
    BeamEntry* base = entries_.data() + lo;
    std::memmove(base + 1, base, (new_size - lo - 1) * sizeof(BeamEntry));
    *base = BeamEntry{distance, id};
    size_ = new_size;
    cursor_ = std::min(cursor_, lo);
  }

  bool has_unexpanded() const noexcept { return cursor_ < size_; }

  std::uint32_t expand_next() noexcept {
    BeamEntry& entry = entries_[cursor_];
    const std::uint32_t id = entry.tagged_id;
    entry.tagged_id |= kExpanded;
    do {
      ++cursor_;
    } while (cursor_ < size_ && (entries_[cursor_].tagged_id & kExpanded) != 0);
    return id;
  }

  // Valid only while has_unexpanded(); used to prefetch the next link list.
  std::uint32_t peek_unexpanded() const noexcept { return entries_[cursor_].tagged_id; }

  std::size_t size() const noexcept { return size_; }
  std::uint32_t id(std::size_t i) const noexcept { return entries_[i].tagged_id & ~kExpanded; }
  float distance(std::size_t i) const noexcept { return entries_[i].distance; }

 private:
  std::vector<BeamEntry> entries_;
  std::size_t ef_ = 0;
  std::size_t size_ = 0;
  std::size_t cursor_ = 0;
};

}  // namespace annvil::detail
