#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "annvil/pq.hpp"

namespace annvil {

enum class Metric : std::uint32_t {
  L2 = 0,       // Squared Euclidean distance.
  Angular = 1,  // 1 - cosine similarity. Vectors are normalized on the way in.
};

struct IndexOptions {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t capacity = 0;  // Fixed up front; see Index for why.
  std::uint32_t M = 16;      // Links per node on upper layers; layer 0 gets 2 * M.
  std::uint32_t ef_construction = 200;
  std::uint64_t seed = 42;
  // Algorithm 4's keepPrunedConnections. Off matches hnswlib; see the README
  // for the measured effect.
  bool keep_pruned = false;
};

struct Neighbor {
  std::uint64_t label = 0;
  float distance = 0.0f;
};

struct SearchStats {
  std::uint64_t distance_computations = 0;
};

struct MemoryBreakdown {
  std::size_t vectors = 0;
  std::size_t links = 0;  // Layer 0, upper layers, levels and offsets.
  std::size_t labels = 0;
  std::size_t pq = 0;
  std::size_t total() const noexcept { return vectors + links + labels + pq; }
};

struct VerifyReport {
  bool ok = true;
  std::vector<std::string> errors;  // Capped so a badly damaged file stays readable.
};

// HNSW index (Malkov & Yashunin, TPAMI 2020).
//
// The in-memory form and the on-disk form are the same flat arrays, so `open` maps
// a file and searches it with no deserialization. Node levels are drawn from a
// hash of (seed, id) when the index is created, which fixes the size of every array
// up front. That removes reallocation from the write path, and with it the only
// reason a reader would need to take a lock.
//
// Thread safety: `add` may run from many threads at once, and `search` may run
// concurrently with `add` and with itself. Readers never block. A reader racing a
// writer may miss the newest points or follow a stale edge, but it only ever
// dereferences fully written nodes. `save` needs writers quiesced. `attach_pq` and
// `reorder` replace storage, so nothing else may touch the index while they run.
class Index {
 public:
  explicit Index(const IndexOptions& options);
  Index(Index&&) noexcept;
  Index& operator=(Index&&) noexcept;
  ~Index();

  // Maps `path` read-only. Header and section sizes are always validated; pass
  // `verify` to also check every link and the payload checksum before use.
  static Index open(const std::filesystem::path& path, bool verify = false);

  // Writes to a sibling temporary file and renames it, so a crash mid-save leaves
  // any previous file at `path` intact.
  void save(const std::filesystem::path& path) const;

  void add(std::span<const float> vector, std::uint64_t label);
  // Inserts `n` vectors of `dim()` floats. Labels default to 0..n-1.
  void add_batch(std::span<const float> vectors, std::size_t threads,
                 std::span<const std::uint64_t> labels = {});

  // Returns how many results were written to `out`, closest first. `ef` is raised
  // to at least `out.size()`.
  std::size_t search(std::span<const float> query, std::size_t ef, std::span<Neighbor> out,
                     SearchStats* stats = nullptr) const;

  // Trains a quantizer on the stored vectors and encodes all of them.
  void attach_pq(const PqTrainOptions& options);
  // Walks the graph on PQ codes alone, then rescores the best `rerank` beam entries
  // with the full vectors. rerank = 0 returns approximate distances and never
  // touches the vector section, so it can stay paged out.
  std::size_t search_pq(std::span<const float> query, std::size_t ef, std::size_t rerank,
                        std::span<Neighbor> out, SearchStats* stats = nullptr) const;

  // Renumbers nodes breadth-first from the entry point on layer 0, so that nodes
  // visited together sit close in memory. Results are unchanged because the graph is
  // isomorphic. Unmaps the old storage: no concurrent readers or writers.
  void reorder();

  // Cache lines prefetched per vector during search. The default covers the whole
  // vector; this exists to measure that choice and to tune it on other hardware.
  // Results never change. Not synchronized with concurrent searches.
  void set_prefetch_lines(std::size_t lines) noexcept;

  VerifyReport verify() const;

  std::size_t size() const noexcept;
  std::size_t capacity() const noexcept;
  std::size_t dim() const noexcept;
  Metric metric() const noexcept;
  std::uint32_t M() const noexcept;
  std::uint32_t ef_construction() const noexcept;
  int max_level() const noexcept;
  bool read_only() const noexcept;
  bool has_pq() const noexcept;
  std::size_t pq_m() const noexcept;
  MemoryBreakdown memory() const noexcept;

  // Introspection for tests and tooling.
  std::vector<std::uint32_t> neighbors(std::uint32_t node, int level) const;
  int node_level(std::uint32_t node) const;
  // The node every search starts from. Throws std::out_of_range on an empty index.
  std::uint32_t entry_node() const;

  // Defined in src/detail; public only so the implementation's free functions can name it.
  struct Impl;

 private:
  explicit Index(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace annvil
