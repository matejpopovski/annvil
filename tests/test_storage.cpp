#include <doctest/doctest.h>

#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "annvil/index.hpp"
#include "detail/format.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 16;

Index small_index(std::size_t n = 800, Metric metric = Metric::L2) {
  const auto data = clustered(n, kDim, 31);
  IndexOptions o;
  o.dim = kDim;
  o.metric = metric;
  o.capacity = n + 50;  // Spare capacity must not leak into the saved layout.
  o.M = 8;
  o.ef_construction = 64;
  Index index(o);
  index.add_batch(data, 1);
  return index;
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Rewrites a header field and its checksum, modelling a file that is well-formed at the
// checksum level but lies about its contents.
template <class Mutate>
void patch_header(std::vector<char>& bytes, Mutate&& mutate) {
  detail::FileHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  mutate(header);
  header.header_checksum =
      detail::hash_bytes(&header, offsetof(detail::FileHeader, header_checksum));
  std::memcpy(bytes.data(), &header, sizeof(header));
}

}  // namespace

TEST_CASE("a saved index returns identical results when mapped back") {
  for (const Metric metric : {Metric::L2, Metric::Angular}) {
    TempDir dir;
    const Index original = small_index(800, metric);
    original.save(dir.file("a.annvil"));
    const Index mapped = Index::open(dir.file("a.annvil"), true);

    CHECK(mapped.read_only());
    CHECK(mapped.size() == original.size());
    CHECK(mapped.dim() == original.dim());
    CHECK(mapped.metric() == original.metric());
    CHECK(mapped.M() == original.M());
    CHECK(mapped.ef_construction() == original.ef_construction());
    CHECK(mapped.max_level() == original.max_level());

    const auto queries = clustered(50, kDim, 32);
    for (std::size_t q = 0; q < 50; ++q) {
      std::vector<Neighbor> a(10), b(10);
      const auto query = std::span<const float>(queries).subspan(q * kDim, kDim);
      REQUIRE(original.search(query, 50, a) == 10);
      REQUIRE(mapped.search(query, 50, b) == 10);
      for (std::size_t i = 0; i < 10; ++i) {
        CHECK(a[i].label == b[i].label);
        CHECK(a[i].distance == b[i].distance);
      }
    }
  }
}

TEST_CASE("a mapped index refuses writes") {
  TempDir dir;
  small_index().save(dir.file("a.annvil"));
  Index mapped = Index::open(dir.file("a.annvil"));
  const std::vector<float> row(kDim, 0.0f);
  CHECK_THROWS_AS(mapped.add(row, 0), std::logic_error);
  CHECK_THROWS_AS(mapped.reorder(), std::logic_error);
  CHECK_THROWS_AS(mapped.attach_pq(PqTrainOptions{}), std::logic_error);
}

TEST_CASE("an empty index round-trips") {
  TempDir dir;
  IndexOptions o;
  o.dim = kDim;
  o.capacity = 5;
  const Index empty(o);
  empty.save(dir.file("e.annvil"));
  const Index mapped = Index::open(dir.file("e.annvil"), true);
  CHECK(mapped.size() == 0);
  std::vector<Neighbor> out(3);
  const std::vector<float> query(kDim, 0.0f);
  CHECK(mapped.search(query, 10, out) == 0);
}

TEST_CASE("a saved PQ index keeps its codes and codebook") {
  TempDir dir;
  Index index = small_index(1000);
  PqTrainOptions pq;
  pq.m = 4;
  pq.iterations = 5;
  index.attach_pq(pq);
  index.save(dir.file("p.annvil"));
  const Index mapped = Index::open(dir.file("p.annvil"), true);
  REQUIRE(mapped.has_pq());
  CHECK(mapped.pq_m() == 4);
  const auto queries = clustered(20, kDim, 33);
  for (std::size_t q = 0; q < 20; ++q) {
    std::vector<Neighbor> a(5), b(5);
    const auto query = std::span<const float>(queries).subspan(q * kDim, kDim);
    REQUIRE(index.search_pq(query, 50, 20, a) == 5);
    REQUIRE(mapped.search_pq(query, 50, 20, b) == 5);
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(a[i].label == b[i].label);
    }
  }
}

TEST_CASE("reordered indexes survive a save") {
  TempDir dir;
  Index index = small_index();
  index.reorder();
  index.save(dir.file("r.annvil"));
  CHECK(Index::open(dir.file("r.annvil"), true).verify().ok);
}

TEST_CASE("overwriting a file through save is atomic") {
  TempDir dir;
  const auto path = dir.file("a.annvil");
  small_index(300).save(path);
  small_index(500).save(path);
  CHECK(Index::open(path, true).size() == 500);
  CHECK_FALSE(std::filesystem::exists(dir.file("a.annvil.tmp")));
}

TEST_CASE("open rejects files that are not valid indexes") {
  TempDir dir;
  const auto path = dir.file("bad.annvil");
  CHECK_THROWS_AS(Index::open(dir.file("missing.annvil")), std::runtime_error);

  write_file(path, std::vector<char>(10, 'x'));
  CHECK_THROWS_AS(Index::open(path), std::runtime_error);

  write_file(path, std::vector<char>(8192, 'x'));
  CHECK_THROWS_AS(Index::open(path), std::runtime_error);

  small_index().save(path);
  const std::vector<char> good = read_file(path);

  SUBCASE("truncated at every block boundary") {
    for (std::size_t size = 4096; size < good.size(); size += good.size() / 7 + 1) {
      std::vector<char> cut(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(size));
      write_file(path, cut);
      CHECK_THROWS_AS(Index::open(path), std::runtime_error);
    }
  }
  SUBCASE("a flipped header bit fails the checksum") {
    std::vector<char> bad = good;
    bad[24] ^= 1;
    write_file(path, bad);
    CHECK_THROWS_WITH_AS(Index::open(path), doctest::Contains("checksum"), std::runtime_error);
  }
  SUBCASE("a future version is refused") {
    std::vector<char> bad = good;
    patch_header(bad, [](detail::FileHeader& h) { h.version = 99; });
    write_file(path, bad);
    CHECK_THROWS_WITH_AS(Index::open(path), doctest::Contains("version"), std::runtime_error);
  }
  SUBCASE("a checksummed header with absurd sizes is refused") {
    std::vector<char> bad = good;
    patch_header(bad, [](detail::FileHeader& h) { h.count = std::uint64_t{1} << 40; });
    write_file(path, bad);
    CHECK_THROWS_AS(Index::open(path), std::runtime_error);
  }
  SUBCASE("a checksummed header with an impossible M is refused") {
    std::vector<char> bad = good;
    patch_header(bad, [](detail::FileHeader& h) { h.M = 100000; });
    write_file(path, bad);
    CHECK_THROWS_AS(Index::open(path), std::runtime_error);
  }
  SUBCASE("a checksummed header with an out-of-range entry point is refused") {
    std::vector<char> bad = good;
    patch_header(bad, [](detail::FileHeader& h) { h.entry = 1u << 30; });
    write_file(path, bad);
    CHECK_THROWS_AS(Index::open(path), std::runtime_error);
  }
}

TEST_CASE("verify catches payload corruption that a lazy open cannot") {
  TempDir dir;
  const auto path = dir.file("c.annvil");
  small_index().save(path);
  std::vector<char> bytes = read_file(path);
  detail::FileHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  const detail::Layout layout = detail::compute_layout({.nodes = header.count,
                                                        .link0_slots = 1 + header.M0,
                                                        .upper_slots = header.upper_slots,
                                                        .stride = header.stride,
                                                        .dim = header.dim,
                                                        .pq_m = 0});

  SUBCASE("a link aimed out of range") {
    const std::uint32_t bogus = 0x7fffffff;
    std::memcpy(bytes.data() + layout.links0.offset + sizeof(std::uint32_t), &bogus, sizeof(bogus));
  }
  SUBCASE("a bit flip in a vector") {
    bytes[layout.vectors.offset + 100] ^= 0x40;
  }
  SUBCASE("a link count above the cap") {
    const std::uint32_t bogus = 1000;
    std::memcpy(bytes.data() + layout.links0.offset, &bogus, sizeof(bogus));
  }

  write_file(path, bytes);
  const Index lazy = Index::open(path);
  CHECK_FALSE(lazy.verify().ok);
  CHECK_THROWS_AS(Index::open(path, true), std::runtime_error);
}

TEST_CASE("random single-byte corruption is detected and never crashes verify") {
  TempDir dir;
  const auto path = dir.file("f.annvil");
  small_index(200).save(path);
  const std::vector<char> good = read_file(path);
  std::mt19937_64 rng(77);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<char> bad = good;
    const std::size_t at = rng() % bad.size();
    const auto flip = static_cast<char>(1 << (rng() % 8));
    bad[at] = static_cast<char>(bad[at] ^ flip);
    write_file(path, bad);
    bool rejected = false;
    try {
      rejected = !Index::open(path).verify().ok;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    // Padding between sections is outside every checksum and harmless.
    detail::FileHeader header;
    std::memcpy(&header, good.data(), sizeof(header));
    const detail::Layout layout = detail::compute_layout({.nodes = header.count,
                                                          .link0_slots = 1 + header.M0,
                                                          .upper_slots = header.upper_slots,
                                                          .stride = header.stride,
                                                          .dim = header.dim,
                                                          .pq_m = 0});
    bool in_payload = at < sizeof(header);
    for (const detail::Section& s : {layout.levels, layout.upper_offsets, layout.links0,
                                     layout.upper, layout.labels, layout.vectors}) {
      in_payload = in_payload || (at >= s.offset && at < s.offset + s.size);
    }
    if (in_payload) {
      CHECK(rejected);
    }
  }
}
