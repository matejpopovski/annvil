#include "annvil/dataset.hpp"

#include <cstring>
#include <stdexcept>

#include "detail/region.hpp"

namespace annvil {
namespace {

constexpr char kDatasetMagic[8] = {'A', 'N', 'N', 'B', '0', '0', '0', '1'};
constexpr std::size_t kDatasetHeaderSize = 64;

struct DatasetHeader {
  char magic[8];
  std::uint32_t metric;
  std::uint32_t dim;
  std::uint64_t train_count;
  std::uint64_t test_count;
  std::uint64_t gt_k;
};

}  // namespace

struct Dataset::Mapping {
  detail::Region region;
};

Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;
Dataset::~Dataset() = default;

Dataset Dataset::open(const std::filesystem::path& path) {
  Dataset dataset;
  dataset.mapping_ = std::make_unique<Mapping>();
  dataset.mapping_->region = detail::Region::map_file(path);
  const detail::Region& region = dataset.mapping_->region;
  if (region.size() < kDatasetHeaderSize) {
    throw std::runtime_error(path.string() + ": too small to be a dataset");
  }
  DatasetHeader header;
  std::memcpy(&header, region.data(), sizeof(header));
  if (std::memcmp(header.magic, kDatasetMagic, sizeof(header.magic)) != 0) {
    throw std::runtime_error(path.string() + ": not a .annb dataset");
  }
  if (header.metric > 1 || header.dim == 0 || header.dim > (1u << 20) ||
      header.train_count > (std::uint64_t{1} << 32) ||
      header.test_count > (std::uint64_t{1} << 32) || header.gt_k > (1u << 20)) {
    throw std::runtime_error(path.string() + ": implausible dataset header");
  }
  const std::uint64_t train_bytes = header.train_count * header.dim * sizeof(float);
  const std::uint64_t test_bytes = header.test_count * header.dim * sizeof(float);
  const std::uint64_t gt_bytes = header.test_count * header.gt_k * sizeof(std::int32_t);
  if (region.size() != kDatasetHeaderSize + train_bytes + test_bytes + 2 * gt_bytes) {
    throw std::runtime_error(path.string() + ": file size does not match its header");
  }

  dataset.metric_ = static_cast<Metric>(header.metric);
  dataset.dim_ = header.dim;
  dataset.train_count_ = header.train_count;
  dataset.test_count_ = header.test_count;
  dataset.gt_k_ = header.gt_k;
  const std::byte* cursor = region.data() + kDatasetHeaderSize;
  dataset.train_ = reinterpret_cast<const float*>(cursor);
  cursor += train_bytes;
  dataset.test_ = reinterpret_cast<const float*>(cursor);
  cursor += test_bytes;
  dataset.gt_ids_ = reinterpret_cast<const std::int32_t*>(cursor);
  cursor += gt_bytes;
  dataset.gt_dists_ = reinterpret_cast<const float*>(cursor);
  return dataset;
}

}  // namespace annvil
