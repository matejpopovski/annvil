#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "annvil/index.hpp"
#include "detail/index_impl.hpp"

namespace annvil {
namespace {

using detail::FileHeader;
using detail::Layout;
using detail::Section;

constexpr std::size_t kMaxReportedErrors = 16;

[[noreturn]] void bad_file(const std::filesystem::path& path, const std::string& reason) {
  throw std::runtime_error(path.string() + ": " + reason);
}

// Removes the temporary file unless the write reaches the final rename.
class TempFile {
 public:
  explicit TempFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd_ < 0) {
      throw_errno("cannot create");
    }
  }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  ~TempFile() {
    if (fd_ >= 0) {
      ::close(fd_);
      ::unlink(path_.c_str());
    }
  }

  void write_at(std::uint64_t offset, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    while (size > 0) {
      // macOS rejects single writes of 2 GiB or more.
      const std::size_t chunk = std::min<std::size_t>(size, std::size_t{1} << 30);
      const ssize_t written = ::pwrite(fd_, bytes, chunk, static_cast<off_t>(offset));
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw_errno("cannot write");
      }
      bytes += written;
      offset += static_cast<std::uint64_t>(written);
      size -= static_cast<std::size_t>(written);
    }
  }

  void commit(const std::filesystem::path& destination, std::uint64_t total_size) {
    if (::ftruncate(fd_, static_cast<off_t>(total_size)) != 0 || ::fsync(fd_) != 0) {
      throw_errno("cannot flush");
    }
    ::close(fd_);
    fd_ = -1;
    if (::rename(path_.c_str(), destination.c_str()) != 0) {
      const int saved = errno;
      ::unlink(path_.c_str());
      errno = saved;
      throw_errno("cannot rename");
    }
  }

 private:
  [[noreturn]] void throw_errno(const char* what) const {
    throw std::runtime_error(std::string(what) + " " + path_.string() + ": " +
                             std::strerror(errno));
  }

  std::filesystem::path path_;
  int fd_ = -1;
};

struct SectionData {
  const void* data;
  std::size_t size;
  Section where;
};

std::vector<SectionData> payload_sections(const Index::Impl& ix, const Layout& layout,
                                          std::size_t n) {
  std::vector<SectionData> out = {
      {ix.levels, static_cast<std::size_t>(layout.levels.size), layout.levels},
      {ix.upper_offsets, static_cast<std::size_t>(layout.upper_offsets.size), layout.upper_offsets},
      {ix.links0, static_cast<std::size_t>(layout.links0.size), layout.links0},
      {ix.upper, static_cast<std::size_t>(layout.upper.size), layout.upper},
      {ix.labels, static_cast<std::size_t>(layout.labels.size), layout.labels},
      {ix.vectors, static_cast<std::size_t>(layout.vectors.size), layout.vectors},
  };
  if (ix.has_pq) {
    const std::span<const float> codebook = ix.pq.codebook();
    out.push_back({codebook.data(), codebook.size_bytes(), layout.pq_codebook});
    out.push_back({ix.codes, n * ix.pq.m(), layout.pq_codes});
  }
  return out;
}

std::uint64_t hash_sections(const std::vector<SectionData>& sections) noexcept {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const SectionData& section : sections) {
    hash = detail::hash_bytes(section.data, section.size, hash);
  }
  return hash;
}

std::size_t used_upper_slots(const Index::Impl& ix, std::size_t n) noexcept {
  if (n == 0) {
    return 0;
  }
  return ix.upper_offsets[n - 1] + static_cast<std::size_t>(ix.levels[n - 1]) * ix.upper_slots;
}

Layout saved_layout(const Index::Impl& ix, std::size_t n) noexcept {
  return detail::compute_layout({.nodes = n,
                                 .link0_slots = ix.link0_slots,
                                 .upper_slots = used_upper_slots(ix, n),
                                 .stride = ix.stride,
                                 .dim = ix.dim,
                                 .pq_m = ix.has_pq ? ix.pq.m() : 0});
}

}  // namespace

void Index::save(const std::filesystem::path& path) const {
  const Impl& ix = *impl_;
  const std::size_t n = size();
  const Layout layout = saved_layout(ix, n);
  const std::vector<SectionData> sections = payload_sections(ix, layout, n);

  FileHeader header{};
  std::memcpy(header.magic, detail::kMagic, sizeof(header.magic));
  header.version = detail::kFormatVersion;
  header.endian_tag = detail::kEndianTag;
  header.dim = static_cast<std::uint32_t>(ix.dim);
  header.stride = static_cast<std::uint32_t>(ix.stride);
  header.metric = static_cast<std::uint32_t>(ix.metric);
  header.M = ix.M;
  header.M0 = ix.M0;
  header.ef_construction = ix.ef_construction;
  header.pq_m = ix.has_pq ? static_cast<std::uint32_t>(ix.pq.m()) : 0;
  header.count = n;
  header.seed = ix.seed;
  header.entry = ix.entry.load(std::memory_order_acquire);
  header.upper_slots = used_upper_slots(ix, n);
  header.level_multiplier = ix.level_multiplier;
  header.file_size = layout.total;
  header.payload_checksum = hash_sections(sections);
  header.header_checksum = detail::hash_bytes(&header, offsetof(FileHeader, header_checksum));

  std::filesystem::path temporary = path;
  temporary += ".tmp";
  TempFile file(temporary);
  file.write_at(0, &header, sizeof(header));
  for (const SectionData& section : sections) {
    file.write_at(section.where.offset, section.data, section.size);
  }
  file.commit(path, layout.total);
}

Index Index::open(const std::filesystem::path& path, bool verify) {
  detail::Region region = detail::Region::map_file(path);
  if (region.size() < sizeof(FileHeader)) {
    bad_file(path, "too small to be an annvil index");
  }
  FileHeader h;
  std::memcpy(&h, region.data(), sizeof(h));
  if (std::memcmp(h.magic, detail::kMagic, sizeof(h.magic)) != 0) {
    bad_file(path, "not an annvil index");
  }
  if (h.endian_tag != detail::kEndianTag) {
    bad_file(path, "written on a machine of different endianness");
  }
  if (h.version != detail::kFormatVersion) {
    bad_file(path, "unsupported format version " + std::to_string(h.version));
  }
  if (h.header_checksum != detail::hash_bytes(&h, offsetof(FileHeader, header_checksum))) {
    bad_file(path, "header checksum mismatch");
  }

  // The checksum proves the header is what was written, not that a crafted file is
  // sane, and a corrupt size here would turn into wild reads through the mapping.
  const bool valid_shape = h.dim >= 1 && h.dim <= kMaxDim &&
                           h.stride == detail::align_up(h.dim, 16) && h.M >= 2 && h.M <= kMaxM &&
                           h.M0 == 2 * h.M && h.metric <= 1 && h.count <= kMaxCapacity &&
                           h.upper_slots <= 0xffffffffULL && h.pq_m <= h.dim;
  if (!valid_shape) {
    bad_file(path, "header describes an invalid index");
  }
  const bool valid_entry = h.count == 0
                               ? h.entry == kNoEntry
                               : entry_id(h.entry) < h.count && entry_level(h.entry) >= 0 &&
                                     entry_level(h.entry) <= kMaxLevel;
  if (!valid_entry) {
    bad_file(path, "header has an invalid entry point");
  }
  const Layout layout = detail::compute_layout({.nodes = h.count,
                                                .link0_slots = std::uint64_t{1} + h.M0,
                                                .upper_slots = h.upper_slots,
                                                .stride = h.stride,
                                                .dim = h.dim,
                                                .pq_m = h.pq_m});
  if (layout.total != h.file_size || layout.total != region.size()) {
    bad_file(path, "file size does not match its header (truncated?)");
  }

  auto impl = std::make_unique<Impl>();
  Impl& ix = *impl;
  ix.dim = h.dim;
  ix.kernel_dim = detail::align_up(h.dim, 4);
  ix.set_stride(h.stride);
  ix.metric = static_cast<Metric>(h.metric);
  ix.M = h.M;
  ix.M0 = h.M0;
  ix.ef_construction = h.ef_construction;
  ix.seed = h.seed;
  ix.level_multiplier = h.level_multiplier;
  ix.capacity = h.count;
  ix.link0_slots = std::size_t{1} + h.M0;
  ix.upper_slots = std::size_t{1} + h.M;
  ix.total_upper_slots = h.upper_slots;
  ix.read_only = true;
  ix.payload_checksum = h.payload_checksum;
  ix.entry.store(h.entry, std::memory_order_relaxed);
  ix.next_id.store(static_cast<std::uint32_t>(h.count), std::memory_order_relaxed);
  ix.published.store(static_cast<std::uint32_t>(h.count), std::memory_order_relaxed);
  ix.bind(std::move(region), layout);

  if (h.pq_m != 0) {
    const auto* first =
        reinterpret_cast<const float*>(ix.region.data() + layout.pq_codebook.offset);
    ix.pq = ProductQuantizer::from_codebook(
        h.dim, h.pq_m, std::vector<float>(first, first + layout.pq_codebook.size / sizeof(float)));
    ix.codes = reinterpret_cast<const std::uint8_t*>(ix.region.data() + layout.pq_codes.offset);
    ix.has_pq = true;
  }

  Index index(std::move(impl));
  if (verify) {
    const VerifyReport report = index.verify();
    if (!report.ok) {
      bad_file(path, "verification failed: " + report.errors.front());
    }
  }
  return index;
}

VerifyReport Index::verify() const {
  const Impl& ix = *impl_;
  const std::size_t n = size();
  VerifyReport report;
  const auto fail = [&report](std::string message) {
    report.ok = false;
    if (report.errors.size() < kMaxReportedErrors) {
      report.errors.push_back(std::move(message));
    }
  };

  if (ix.read_only) {
    const Layout layout = saved_layout(ix, n);
    if (hash_sections(payload_sections(ix, layout, n)) != ix.payload_checksum) {
      fail("payload checksum mismatch");
    }
  }

  // Levels and offsets first: the link checks below index through them.
  std::uint64_t expected_offset = 0;
  int highest = -1;
  for (std::size_t id = 0; id < n; ++id) {
    if (ix.levels[id] > kMaxLevel) {
      fail("node " + std::to_string(id) + " has level " + std::to_string(ix.levels[id]));
      return report;
    }
    if (ix.upper_offsets[id] != expected_offset) {
      fail("node " + std::to_string(id) + " has a misplaced upper-layer offset");
      return report;
    }
    expected_offset += static_cast<std::uint64_t>(ix.levels[id]) * ix.upper_slots;
    highest = std::max<int>(highest, ix.levels[id]);
  }
  if (expected_offset > ix.total_upper_slots) {
    fail("upper-layer lists overrun their section");
    return report;
  }

  std::vector<std::uint32_t> seen;
  for (std::size_t id = 0; id < n; ++id) {
    const auto node = static_cast<std::uint32_t>(id);
    for (int level = 0; level <= ix.levels[id]; ++level) {
      const std::uint32_t* list = ix.list(node, level);
      const std::uint32_t count = load_slot(list[0]);
      if (count > ix.max_links(level)) {
        fail("node " + std::to_string(id) + " level " + std::to_string(level) + " has " +
             std::to_string(count) + " links");
        continue;
      }
      seen.clear();
      for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t neighbor = load_slot(list[1 + i]);
        if (neighbor >= n) {
          fail("node " + std::to_string(id) + " links to out-of-range id " +
               std::to_string(neighbor));
        } else if (neighbor == node) {
          fail("node " + std::to_string(id) + " links to itself");
        } else if (ix.levels[neighbor] < level) {
          fail("node " + std::to_string(id) + " links to " + std::to_string(neighbor) +
               ", which is absent from level " + std::to_string(level));
        }
        seen.push_back(neighbor);
      }
      std::sort(seen.begin(), seen.end());
      if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
        fail("node " + std::to_string(id) + " has duplicate links on level " +
             std::to_string(level));
      }
    }
  }

  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (n > 0) {
    if (packed == kNoEntry || entry_id(packed) >= n) {
      fail("entry point is out of range");
    } else if (ix.levels[entry_id(packed)] != entry_level(packed) ||
               entry_level(packed) != highest) {
      fail("entry point is not on the top level");
    }
  }
  return report;
}

}  // namespace annvil
