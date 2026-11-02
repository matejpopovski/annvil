#include <doctest/doctest.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "annvil/dataset.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

struct Header {
  char magic[8] = {'A', 'N', 'N', 'B', '0', '0', '0', '1'};
  std::uint32_t metric = 1;
  std::uint32_t dim = 2;
  std::uint64_t train = 3;
  std::uint64_t test = 1;
  std::uint64_t gt_k = 2;
  char padding[24] = {};
};
static_assert(sizeof(Header) == 64);

void write_dataset(const std::filesystem::path& path, const Header& header, std::size_t drop = 0) {
  const std::vector<float> train = {1, 2, 3, 4, 5, 6};
  const std::vector<float> test = {7, 8};
  const std::vector<std::int32_t> ids = {2, 0};
  const std::vector<float> dists = {0.5f, 0.75f};
  std::vector<char> bytes(sizeof(Header) + 24 + 8 + 8 + 8);
  char* cursor = bytes.data();
  const auto put = [&cursor](const void* data, std::size_t size) {
    std::memcpy(cursor, data, size);
    cursor += size;
  };
  put(&header, sizeof(header));
  put(train.data(), 24);
  put(test.data(), 8);
  put(ids.data(), 8);
  put(dists.data(), 8);
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - drop));
}

}  // namespace

TEST_CASE("a dataset file maps to the sections it declares") {
  TempDir dir;
  write_dataset(dir.file("d.annb"), Header{});
  const Dataset d = Dataset::open(dir.file("d.annb"));
  CHECK(d.metric() == Metric::Angular);
  CHECK(d.dim() == 2);
  CHECK(d.train_count() == 3);
  CHECK(d.test_count() == 1);
  CHECK(d.gt_k() == 2);
  CHECK(d.train_vector(2)[1] == 6.0f);
  CHECK(d.test_vector(0)[0] == 7.0f);
  CHECK(d.gt_neighbors(0)[0] == 2);
  CHECK(d.gt_distances(0)[1] == 0.75f);
}

TEST_CASE("malformed dataset files are rejected") {
  TempDir dir;
  const auto path = dir.file("bad.annb");
  SUBCASE("truncated") {
    write_dataset(path, Header{}, 4);
  }
  SUBCASE("bad magic") {
    Header h;
    h.magic[0] = 'X';
    write_dataset(path, h);
  }
  SUBCASE("unknown metric") {
    Header h;
    h.metric = 9;
    write_dataset(path, h);
  }
  SUBCASE("size overflow bait") {
    Header h;
    h.train = ~std::uint64_t{0};
    write_dataset(path, h);
  }
  CHECK_THROWS_AS(Dataset::open(path), std::runtime_error);
}
