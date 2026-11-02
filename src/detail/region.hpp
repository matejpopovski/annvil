// A page-aligned byte range owned through mmap. Build-time storage is an anonymous
// mapping and read-time storage is a file mapping; the search code cannot tell them
// apart, which is what keeps the two paths from drifting.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace annvil::detail {

class Region {
 public:
  Region() = default;
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  Region(Region&& other) noexcept
      : base_(std::exchange(other.base_, nullptr)), size_(std::exchange(other.size_, 0)) {}
  Region& operator=(Region&& other) noexcept {
    if (this != &other) {
      reset();
      base_ = std::exchange(other.base_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  ~Region() { reset(); }

  // Zero-filled and committed on first touch, so a generous capacity costs nothing
  // until points arrive.
  static Region anonymous(std::size_t size) {
    void* base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
      throw std::runtime_error("mmap of " + std::to_string(size) +
                               " bytes failed: " + std::strerror(errno));
    }
    return Region(base, size);
  }

  static Region map_file(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      throw std::runtime_error("cannot open " + path.string() + ": " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("cannot map empty or unreadable file " + path.string());
    }
    const auto size = static_cast<std::size_t>(st.st_size);
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    const int map_errno = errno;
    ::close(fd);
    if (base == MAP_FAILED) {
      throw std::runtime_error("mmap of " + path.string() + " failed: " + std::strerror(map_errno));
    }
    return Region(base, size);
  }

  void reset() noexcept {
    if (base_ != nullptr) {
      ::munmap(base_, size_);
      base_ = nullptr;
      size_ = 0;
    }
  }

  std::byte* data() const noexcept { return static_cast<std::byte*>(base_); }
  std::size_t size() const noexcept { return size_; }

 private:
  Region(void* base, std::size_t size) : base_(base), size_(size) {}

  void* base_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace annvil::detail
