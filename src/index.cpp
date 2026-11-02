#include "annvil/index.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "annvil/kernels.hpp"
#include "annvil/parallel.hpp"
#include "detail/index_impl.hpp"
#include "detail/search_state.hpp"

namespace annvil {
namespace {

using detail::Beam;
using detail::VisitedSet;

struct Candidate {
  float distance;
  std::uint32_t id;
  bool operator<(const Candidate& other) const noexcept { return distance < other.distance; }
};

// Kept per thread rather than pooled: a pool needs a lock on every query, and the
// state is only a bitset and a few small arrays.
struct Scratch {
  VisitedSet visited;
  Beam beam;
  std::vector<float> query;
  std::vector<float> table;
  std::vector<Candidate> found;
  std::vector<Candidate> candidates;
  std::vector<Candidate> selected;
  std::vector<Candidate> pruned;
  std::vector<std::vector<Candidate>> picks;  // Per level, between search and linking.
};

Scratch& thread_scratch() {
  thread_local Scratch scratch;
  return scratch;
}

std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Level is a pure function of (seed, id), not of a shared RNG, so it can be tabulated
// before the first insert and does not depend on thread interleaving.
int draw_level(std::uint64_t seed, std::uint32_t id, double multiplier) noexcept {
  const std::uint64_t bits = splitmix64(seed ^ (std::uint64_t{id} * 0x9e3779b97f4a7c15ULL));
  const double uniform = static_cast<double>((bits >> 11) + 1) * 0x1.0p-53;
  const double level = std::floor(-std::log(uniform) * multiplier);
  return level >= kMaxLevel ? kMaxLevel : static_cast<int>(level);
}

void normalize(float* v, std::size_t dim) noexcept {
  const float norm_sq = kernels::dot(v, v, dim);
  if (norm_sq > 0.0f) {
    const float scale = 1.0f / std::sqrt(norm_sq);
    for (std::size_t i = 0; i < dim; ++i) {
      v[i] *= scale;
    }
  }
}

void require_finite(std::span<const float> v, const char* what) {
  for (const float x : v) {
    if (!std::isfinite(x)) {
      throw std::invalid_argument(std::string(what) + " contains a non-finite value");
    }
  }
}

template <Metric kMetric>
float vector_distance(const float* a, const float* b, std::size_t n) noexcept {
  if constexpr (kMetric == Metric::L2) {
    return kernels::l2_sq(a, b, n);
  } else {
    return 1.0f - kernels::dot(a, b, n);
  }
}

// A distance oracle answers "how far is node i from the query". The search routines
// are templated on it so exact and PQ traversal share one implementation and still
// inline their distance computation.
template <Metric kMetric>
class ExactOracle {
 public:
  ExactOracle(const Index::Impl& index, const float* query) noexcept
      : query_(query),
        vectors_(index.vectors),
        stride_(index.stride),
        n_(index.kernel_dim),
        lines_(index.prefetch_lines) {}

  float distance(std::uint32_t id) const noexcept {
    return vector_distance<kMetric>(query_, vectors_ + id * stride_, n_);
  }
  // The whole vector, not just its first line. On sift-128 this alone doubles QPS,
  // because each vector spans four 128-byte lines and a later line otherwise stalls
  // the kernel partway through (see "Where the time goes" in the README).
  void prefetch(std::uint32_t id) const noexcept {
    const char* bytes = reinterpret_cast<const char*>(vectors_ + id * stride_);
    for (std::size_t line = 0; line < lines_; ++line) {
      __builtin_prefetch(bytes + line * kCacheLine);
    }
  }

 private:
  const float* query_;
  const float* vectors_;
  std::size_t stride_;
  std::size_t n_;
  std::size_t lines_;
};

template <Metric kMetric>
class PqOracle {
 public:
  PqOracle(const std::uint8_t* codes, std::size_t m, const float* table) noexcept
      : codes_(codes), m_(m), table_(table) {}

  float distance(std::uint32_t id) const noexcept {
    const std::uint8_t* code = codes_ + id * m_;
    float sum = 0.0f;
    for (std::size_t j = 0; j < m_; ++j) {
      sum += table_[j * kPqCentroids + code[j]];
    }
    if constexpr (kMetric == Metric::L2) {
      return sum;
    } else {
      return 1.0f - sum;
    }
  }
  void prefetch(std::uint32_t id) const noexcept { __builtin_prefetch(codes_ + id * m_); }

 private:
  const std::uint8_t* codes_;
  std::size_t m_;
  const float* table_;
};

// Wrapping the oracle keeps instrumentation out of the uninstrumented loop entirely,
// rather than leaving a branch on a stats pointer in it.
template <class Oracle>
class Counting {
 public:
  explicit Counting(const Oracle& inner) noexcept : inner_(inner) {}
  float distance(std::uint32_t id) const noexcept {
    ++count_;
    return inner_.distance(id);
  }
  void prefetch(std::uint32_t id) const noexcept { inner_.prefetch(id); }
  std::uint64_t count() const noexcept { return count_; }

 private:
  Oracle inner_;
  mutable std::uint64_t count_ = 0;
};

template <class Oracle>
void greedy_descend(const Index::Impl& ix, const Oracle& oracle, std::uint32_t& node,
                    float& distance, int level) {
  bool improved = true;
  while (improved) {
    improved = false;
    const std::uint32_t* list = ix.list(node, level);
    const std::uint32_t count = std::min(load_slot(list[0]), ix.M);
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t id = load_slot(list[1 + i]);
      const float d = oracle.distance(id);
      if (d < distance) {
        distance = d;
        node = id;
        improved = true;
      }
    }
  }
}

// Algorithm 2 of the paper. On return the beam holds the ef closest nodes found.
template <class Oracle>
void search_layer(const Index::Impl& ix, const Oracle& oracle, std::uint32_t entry,
                  float entry_distance, std::size_t ef, int level, Scratch& scratch) {
  VisitedSet& visited = scratch.visited;
  Beam& beam = scratch.beam;
  visited.prepare(ix.capacity);
  beam.reset(ef);
  beam.insert(entry_distance, entry);
  visited.insert(entry);

  const std::uint32_t max_links = ix.max_links(level);
  std::array<std::uint32_t, kMaxM0> pending;
  while (beam.has_unexpanded()) {
    const std::uint32_t current = beam.expand_next();
    if (beam.has_unexpanded()) {
      // Both ends, because a list is not line-aligned and can straddle two lines.
      const std::uint32_t* next = ix.list(beam.peek_unexpanded(), level);
      __builtin_prefetch(next);
      __builtin_prefetch(next + max_links);
    }
    const std::uint32_t* list = ix.list(current, level);
    const std::uint32_t count = std::min(load_slot(list[0]), max_links);

    // Filter first and compute second, so that every vector is prefetched before
    // the first distance stalls on a miss.
    std::uint32_t unseen = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t id = load_slot(list[1 + i]);
      if (visited.insert(id)) {
        pending[unseen++] = id;
        oracle.prefetch(id);
      }
    }
    for (std::uint32_t i = 0; i < unseen; ++i) {
      const float d = oracle.distance(pending[i]);
      if (d < beam.bound()) {
        beam.insert(d, pending[i]);
      }
    }
  }
}

template <class Oracle>
void traverse(const Index::Impl& ix, const Oracle& oracle, std::uint64_t packed_entry,
              std::size_t ef, Scratch& scratch) {
  std::uint32_t node = entry_id(packed_entry);
  float distance = oracle.distance(node);
  for (int level = entry_level(packed_entry); level > 0; --level) {
    greedy_descend(ix, oracle, node, distance, level);
  }
  search_layer(ix, oracle, node, distance, ef, 0, scratch);
}

// Algorithm 4: a candidate is kept only if it is closer to the base point than to
// every neighbour already kept. This is what preserves long edges between clusters.
// `candidates` must be sorted by distance to the base point.
template <Metric kMetric>
void select_neighbors(const Index::Impl& ix, std::span<const Candidate> candidates,
                      std::size_t limit, Scratch& scratch) {
  std::vector<Candidate>& selected = scratch.selected;
  std::vector<Candidate>& pruned = scratch.pruned;
  selected.clear();
  pruned.clear();
  // With fewer candidates than slots there is nothing to trade off, and pruning
  // would only thin out the sparse early graph.
  if (candidates.size() < limit) {
    selected.assign(candidates.begin(), candidates.end());
    return;
  }
  for (const Candidate& candidate : candidates) {
    if (selected.size() >= limit) {
      break;
    }
    const float* vector = ix.vector(candidate.id);
    bool keep = true;
    for (const Candidate& kept : selected) {
      if (vector_distance<kMetric>(vector, ix.vector(kept.id), ix.kernel_dim) <
          candidate.distance) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push_back(candidate);
    } else {
      pruned.push_back(candidate);
    }
  }
  if (ix.keep_pruned) {
    for (const Candidate& candidate : pruned) {
      if (selected.size() >= limit) {
        break;
      }
      selected.push_back(candidate);
    }
  }
}

void write_list(std::uint32_t* list, std::span<const Candidate> neighbors) noexcept {
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    store_slot(list[1 + i], neighbors[i].id);
  }
  store_slot(list[0], static_cast<std::uint32_t>(neighbors.size()));
}

template <Metric kMetric>
void link_back(Index::Impl& ix, std::uint32_t target, std::uint32_t added, int level,
               Scratch& scratch) {
  const std::scoped_lock lock(ix.locks[target % Index::Impl::kLockStripes].mutex);
  std::uint32_t* list = ix.list(target, level);
  const std::uint32_t max_links = ix.max_links(level);
  const std::uint32_t count = list[0];
  if (count < max_links) {
    store_slot(list[1 + count], added);
    store_slot(list[0], count + 1);
    return;
  }
  // The list is full, so `added` competes for a place on the same terms as the
  // existing neighbours.
  const float* base = ix.vector(target);
  std::vector<Candidate>& pool = scratch.candidates;
  pool.clear();
  pool.push_back({vector_distance<kMetric>(base, ix.vector(added), ix.kernel_dim), added});
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t id = list[1 + i];
    pool.push_back({vector_distance<kMetric>(base, ix.vector(id), ix.kernel_dim), id});
  }
  std::sort(pool.begin(), pool.end());
  select_neighbors<kMetric>(ix, pool, max_links, scratch);
  write_list(list, scratch.selected);
}

// Algorithm 1, for a node whose vector and label are already stored.
//
// The paper links each level as it descends. Done concurrently, that publishes a node
// on level l+1 before its level l list exists: another insert descending through it
// lands on an empty list and its whole layer search dead-ends. hnswlib prevents this
// by holding the node's lock for the whole insertion, and taking that lock on every
// read during a build. Here readers take no locks, so the order changes instead: all
// levels are searched first, then linked bottom-up. By the time a node is reachable
// on a level, every level below it is complete. It also means a node can never
// reach itself during its own search, and that nobody can link into a list before
// it is written, so writing it needs no lock.
template <Metric kMetric>
void insert_node(Index::Impl& ix, std::uint32_t id) {
  const int level = ix.levels[id];
  std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (packed == kNoEntry) {
    if (ix.entry.compare_exchange_strong(packed, pack_entry(id, level), std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      return;
    }
  }

  Scratch& scratch = thread_scratch();
  const ExactOracle<kMetric> oracle(ix, ix.vector(id));
  std::uint32_t node = entry_id(packed);
  float distance = oracle.distance(node);
  const int top = entry_level(packed);
  for (int lc = top; lc > level; --lc) {
    greedy_descend(ix, oracle, node, distance, lc);
  }

  const int highest_linked = std::min(level, top);
  std::vector<std::vector<Candidate>>& picks = scratch.picks;
  if (picks.size() <= static_cast<std::size_t>(highest_linked)) {
    picks.resize(static_cast<std::size_t>(highest_linked) + 1);
  }
  for (int lc = highest_linked; lc >= 0; --lc) {
    search_layer(ix, oracle, node, distance, ix.ef_construction, lc, scratch);
    std::vector<Candidate>& found = scratch.found;
    found.clear();
    for (std::size_t i = 0; i < scratch.beam.size(); ++i) {
      found.push_back({scratch.beam.distance(i), scratch.beam.id(i)});
    }
    select_neighbors<kMetric>(ix, found, ix.M, scratch);
    picks[static_cast<std::size_t>(lc)] = scratch.selected;
    node = scratch.beam.id(0);
    distance = scratch.beam.distance(0);
  }

  for (int lc = 0; lc <= highest_linked; ++lc) {
    const std::vector<Candidate>& chosen = picks[static_cast<std::size_t>(lc)];
    write_list(ix.list(id, lc), chosen);
    for (const Candidate& neighbor : chosen) {
      link_back<kMetric>(ix, neighbor.id, id, lc, scratch);
    }
  }

  if (level > top) {
    std::uint64_t current = ix.entry.load(std::memory_order_acquire);
    while (entry_level(current) < level &&
           !ix.entry.compare_exchange_weak(current, pack_entry(id, level),
                                           std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
  }
}

const float* prepare_query(const Index::Impl& ix, std::span<const float> query, Scratch& scratch) {
  if (query.size() != ix.dim) {
    throw std::invalid_argument("query has " + std::to_string(query.size()) +
                                " dimensions, index has " + std::to_string(ix.dim));
  }
  require_finite(query, "query");
  scratch.query.assign(ix.kernel_dim, 0.0f);
  std::copy(query.begin(), query.end(), scratch.query.begin());
  if (ix.metric == Metric::Angular) {
    normalize(scratch.query.data(), ix.dim);
  }
  return scratch.query.data();
}

template <class Oracle>
void run_traversal(const Index::Impl& ix, const Oracle& oracle, std::uint64_t packed,
                   std::size_t ef, Scratch& scratch, SearchStats* stats) {
  if (stats != nullptr) {
    const Counting<Oracle> counting(oracle);
    traverse(ix, counting, packed, ef, scratch);
    stats->distance_computations += counting.count();
  } else {
    traverse(ix, oracle, packed, ef, scratch);
  }
}

}  // namespace

Index::Index(const IndexOptions& options) : impl_(std::make_unique<Impl>()) {
  if (options.dim == 0 || options.dim > kMaxDim) {
    throw std::invalid_argument("dim must be in [1, " + std::to_string(kMaxDim) + "]");
  }
  if (options.capacity == 0 || options.capacity > kMaxCapacity) {
    throw std::invalid_argument("capacity must be in [1, 2^31 - 1]");
  }
  if (options.M < 2 || options.M > kMaxM) {
    throw std::invalid_argument("M must be in [2, " + std::to_string(kMaxM) + "]");
  }
  if (options.ef_construction == 0) {
    throw std::invalid_argument("ef_construction must be positive");
  }

  Impl& ix = *impl_;
  ix.dim = options.dim;
  ix.kernel_dim = detail::align_up(options.dim, 4);
  ix.set_stride(detail::align_up(options.dim, 16));
  ix.metric = options.metric;
  ix.M = options.M;
  ix.M0 = 2 * options.M;
  ix.ef_construction = options.ef_construction;
  ix.seed = options.seed;
  ix.level_multiplier = 1.0 / std::log(static_cast<double>(options.M));
  ix.keep_pruned = options.keep_pruned;
  ix.capacity = options.capacity;
  ix.link0_slots = 1 + ix.M0;
  ix.upper_slots = 1 + ix.M;
  ix.locks = std::make_unique<PaddedMutex[]>(Impl::kLockStripes);

  std::vector<std::uint8_t> levels(ix.capacity);
  std::vector<std::uint32_t> offsets(ix.capacity);
  std::uint64_t upper_total = 0;
  for (std::size_t id = 0; id < ix.capacity; ++id) {
    const int level = draw_level(ix.seed, static_cast<std::uint32_t>(id), ix.level_multiplier);
    levels[id] = static_cast<std::uint8_t>(level);
    offsets[id] = static_cast<std::uint32_t>(upper_total);
    upper_total += static_cast<std::uint64_t>(level) * ix.upper_slots;
    if (upper_total > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("capacity too large for 32-bit upper-layer offsets");
    }
  }
  ix.total_upper_slots = upper_total;

  const detail::Layout layout = detail::compute_layout({.nodes = ix.capacity,
                                                        .link0_slots = ix.link0_slots,
                                                        .upper_slots = upper_total,
                                                        .stride = ix.stride,
                                                        .dim = ix.dim,
                                                        .pq_m = 0});
  ix.bind(detail::Region::anonymous(layout.total), layout);
  std::memcpy(ix.levels, levels.data(), levels.size());
  std::memcpy(ix.upper_offsets, offsets.data(), offsets.size() * sizeof(std::uint32_t));
}

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;
Index::~Index() = default;

void Index::add(std::span<const float> vector, std::uint64_t label) {
  Impl& ix = *impl_;
  if (ix.read_only) {
    throw std::logic_error("index was opened read-only");
  }
  if (vector.size() != ix.dim) {
    throw std::invalid_argument("vector has " + std::to_string(vector.size()) +
                                " dimensions, index has " + std::to_string(ix.dim));
  }
  require_finite(vector, "vector");

  std::uint32_t id = ix.next_id.load(std::memory_order_relaxed);
  do {
    if (id >= ix.capacity) {
      throw std::length_error("index is full");
    }
  } while (!ix.next_id.compare_exchange_weak(id, id + 1, std::memory_order_relaxed));

  float* stored = ix.vectors + id * ix.stride;
  std::copy(vector.begin(), vector.end(), stored);
  if (ix.metric == Metric::Angular) {
    normalize(stored, ix.dim);
  }
  ix.labels[id] = label;

  if (ix.metric == Metric::L2) {
    insert_node<Metric::L2>(ix, id);
  } else {
    insert_node<Metric::Angular>(ix, id);
  }
  ix.published.fetch_add(1, std::memory_order_release);
}

void Index::add_batch(std::span<const float> vectors, std::size_t threads,
                      std::span<const std::uint64_t> labels) {
  const std::size_t d = impl_->dim;
  if (vectors.size() % d != 0) {
    throw std::invalid_argument("vector data is not a whole number of rows");
  }
  const std::size_t n = vectors.size() / d;
  if (!labels.empty() && labels.size() != n) {
    throw std::invalid_argument("labels and vectors differ in length");
  }
  parallel_for(n, threads, [&](std::size_t i, std::size_t) {
    add(vectors.subspan(i * d, d), labels.empty() ? i : labels[i]);
  });
}

std::size_t Index::search(std::span<const float> query, std::size_t ef, std::span<Neighbor> out,
                          SearchStats* stats) const {
  const Impl& ix = *impl_;
  const std::size_t k = out.size();
  Scratch& scratch = thread_scratch();
  const float* q = prepare_query(ix, query, scratch);
  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (k == 0 || packed == kNoEntry) {
    return 0;
  }
  ef = std::max({ef, k, std::size_t{1}});
  if (ix.metric == Metric::L2) {
    run_traversal(ix, ExactOracle<Metric::L2>(ix, q), packed, ef, scratch, stats);
  } else {
    run_traversal(ix, ExactOracle<Metric::Angular>(ix, q), packed, ef, scratch, stats);
  }
  const std::size_t found = std::min(k, scratch.beam.size());
  for (std::size_t i = 0; i < found; ++i) {
    out[i] = Neighbor{ix.labels[scratch.beam.id(i)], scratch.beam.distance(i)};
  }
  return found;
}

std::size_t Index::search_pq(std::span<const float> query, std::size_t ef, std::size_t rerank,
                             std::span<Neighbor> out, SearchStats* stats) const {
  const Impl& ix = *impl_;
  if (!ix.has_pq) {
    throw std::logic_error("index has no product quantizer");
  }
  const std::size_t k = out.size();
  Scratch& scratch = thread_scratch();
  const float* q = prepare_query(ix, query, scratch);
  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (k == 0 || packed == kNoEntry) {
    return 0;
  }
  ef = std::max({ef, k, rerank, std::size_t{1}});

  scratch.table.resize(ix.pq.table_size());
  const std::size_t m = ix.pq.m();
  if (ix.metric == Metric::L2) {
    ix.pq.l2_table(q, scratch.table.data());
    run_traversal(ix, PqOracle<Metric::L2>(ix.codes, m, scratch.table.data()), packed, ef, scratch,
                  stats);
  } else {
    ix.pq.dot_table(q, scratch.table.data());
    run_traversal(ix, PqOracle<Metric::Angular>(ix.codes, m, scratch.table.data()), packed, ef,
                  scratch, stats);
  }

  if (rerank == 0) {
    const std::size_t found = std::min(k, scratch.beam.size());
    for (std::size_t i = 0; i < found; ++i) {
      out[i] = Neighbor{ix.labels[scratch.beam.id(i)], scratch.beam.distance(i)};
    }
    return found;
  }

  const std::size_t pool_size = std::min(rerank, scratch.beam.size());
  std::vector<Candidate>& pool = scratch.candidates;
  pool.clear();
  for (std::size_t i = 0; i < pool_size; ++i) {
    const std::uint32_t id = scratch.beam.id(i);
    const float d = ix.metric == Metric::L2
                        ? vector_distance<Metric::L2>(q, ix.vector(id), ix.kernel_dim)
                        : vector_distance<Metric::Angular>(q, ix.vector(id), ix.kernel_dim);
    pool.push_back({d, id});
  }
  const std::size_t found = std::min(k, pool.size());
  std::partial_sort(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(found), pool.end());
  for (std::size_t i = 0; i < found; ++i) {
    out[i] = Neighbor{ix.labels[pool[i].id], pool[i].distance};
  }
  return found;
}

void Index::attach_pq(const PqTrainOptions& options) {
  Impl& ix = *impl_;
  if (ix.read_only) {
    throw std::logic_error("index was opened read-only");
  }
  const std::size_t n = size();
  if (n < kPqCentroids) {
    throw std::invalid_argument("product quantization needs at least 256 stored vectors");
  }
  ProductQuantizer pq = ProductQuantizer::train(ix.vectors, n, ix.stride, ix.dim, options);
  std::vector<std::uint8_t> codes(n * pq.m());
  parallel_for(
      n, options.threads,
      [&](std::size_t i, std::size_t) {
        pq.encode(ix.vectors + i * ix.stride, &codes[i * pq.m()]);
      },
      1024);
  ix.pq = std::move(pq);
  ix.owned_codes = std::move(codes);
  ix.codes = ix.owned_codes.data();
  ix.has_pq = true;
}

void Index::reorder() {
  Impl& ix = *impl_;
  if (ix.read_only) {
    throw std::logic_error("index was opened read-only");
  }
  const std::size_t n = size();
  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (n == 0 || packed == kNoEntry) {
    return;
  }

  constexpr std::uint32_t kUnassigned = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> new_id(n, kUnassigned);
  std::vector<std::uint32_t> order;
  order.reserve(n);
  const auto assign = [&](std::uint32_t old_id) {
    if (new_id[old_id] == kUnassigned) {
      new_id[old_id] = static_cast<std::uint32_t>(order.size());
      order.push_back(old_id);
    }
  };
  assign(entry_id(packed));
  for (std::size_t head = 0; head < order.size(); ++head) {
    const std::uint32_t* list = ix.list(order[head], 0);
    for (std::uint32_t i = 0; i < list[0]; ++i) {
      assign(list[1 + i]);
    }
  }
  // Nodes unreachable from the entry point on layer 0 keep their relative order;
  // they can still be reached through upper-layer links.
  for (std::uint32_t id = 0; id < n; ++id) {
    assign(id);
  }

  const detail::Layout layout = detail::compute_layout({.nodes = ix.capacity,
                                                        .link0_slots = ix.link0_slots,
                                                        .upper_slots = ix.total_upper_slots,
                                                        .stride = ix.stride,
                                                        .dim = ix.dim,
                                                        .pq_m = 0});
  auto next = std::make_unique<Impl>();
  next->bind(detail::Region::anonymous(layout.total), layout);
  next->link0_slots = ix.link0_slots;
  next->upper_slots = ix.upper_slots;

  std::uint32_t cursor = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t old_id = order[i];
    const int level = ix.levels[old_id];
    next->levels[i] = static_cast<std::uint8_t>(level);
    next->upper_offsets[i] = cursor;
    cursor += static_cast<std::uint32_t>(static_cast<std::size_t>(level) * ix.upper_slots);
    next->labels[i] = ix.labels[old_id];
    std::memcpy(next->vectors + i * ix.stride, ix.vector(old_id), ix.stride * sizeof(float));
    for (int lc = 0; lc <= level; ++lc) {
      const std::uint32_t* from = ix.list(old_id, lc);
      std::uint32_t* to = next->list(static_cast<std::uint32_t>(i), lc);
      to[0] = from[0];
      for (std::uint32_t j = 0; j < from[0]; ++j) {
        to[1 + j] = new_id[from[1 + j]];
      }
    }
  }
  // Slots for ids not yet inserted carry over, so the index can keep growing.
  std::memcpy(next->levels + n, ix.levels + n, ix.capacity - n);
  std::memcpy(next->upper_offsets + n, ix.upper_offsets + n,
              (ix.capacity - n) * sizeof(std::uint32_t));

  if (ix.has_pq) {
    const std::size_t m = ix.pq.m();
    std::vector<std::uint8_t> codes(n * m);
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(&codes[i * m], ix.codes + order[i] * m, m);
    }
    ix.owned_codes = std::move(codes);
    ix.codes = ix.owned_codes.data();
  }
  ix.bind(std::move(next->region), layout);
  ix.entry.store(pack_entry(new_id[entry_id(packed)], entry_level(packed)),
                 std::memory_order_release);
}

void Index::set_prefetch_lines(std::size_t lines) noexcept {
  impl_->prefetch_lines = lines;
}

std::size_t Index::size() const noexcept {
  return impl_->published.load(std::memory_order_acquire);
}
std::size_t Index::capacity() const noexcept {
  return impl_->capacity;
}
std::size_t Index::dim() const noexcept {
  return impl_->dim;
}
Metric Index::metric() const noexcept {
  return impl_->metric;
}
std::uint32_t Index::M() const noexcept {
  return impl_->M;
}
std::uint32_t Index::ef_construction() const noexcept {
  return impl_->ef_construction;
}
bool Index::read_only() const noexcept {
  return impl_->read_only;
}
bool Index::has_pq() const noexcept {
  return impl_->has_pq;
}
std::size_t Index::pq_m() const noexcept {
  return impl_->has_pq ? impl_->pq.m() : 0;
}

int Index::max_level() const noexcept {
  const std::uint64_t packed = impl_->entry.load(std::memory_order_acquire);
  return packed == kNoEntry ? -1 : entry_level(packed);
}

MemoryBreakdown Index::memory() const noexcept {
  const Impl& ix = *impl_;
  const std::size_t n = size();
  MemoryBreakdown out;
  out.vectors = n * ix.stride * sizeof(float);
  out.links = n + n * sizeof(std::uint32_t) + n * ix.link0_slots * sizeof(std::uint32_t);
  for (std::size_t id = 0; id < n; ++id) {
    out.links += ix.levels[id] * ix.upper_slots * sizeof(std::uint32_t);
  }
  out.labels = n * sizeof(std::uint64_t);
  if (ix.has_pq) {
    out.pq = n * ix.pq.m() + ix.pq.codebook().size_bytes();
  }
  return out;
}

std::vector<std::uint32_t> Index::neighbors(std::uint32_t node, int level) const {
  const Impl& ix = *impl_;
  if (node >= size() || level < 0 || level > ix.levels[node]) {
    throw std::out_of_range("no such node on that level");
  }
  const std::uint32_t* list = ix.list(node, level);
  const std::uint32_t count = load_slot(list[0]);
  std::vector<std::uint32_t> out(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    out[i] = load_slot(list[1 + i]);
  }
  return out;
}

int Index::node_level(std::uint32_t node) const {
  if (node >= size()) {
    throw std::out_of_range("no such node");
  }
  return impl_->levels[node];
}

std::uint32_t Index::entry_node() const {
  const std::uint64_t packed = impl_->entry.load(std::memory_order_acquire);
  if (packed == kNoEntry) {
    throw std::out_of_range("index is empty");
  }
  return entry_id(packed);
}

}  // namespace annvil
