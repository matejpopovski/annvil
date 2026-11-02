#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace annvil {

// Each subspace is quantized to 256 centroids, so a code is `m` bytes. Only the
// 8-bit case exists: it is what makes the asymmetric lookup a plain byte index.
inline constexpr std::size_t kPqCentroids = 256;

struct PqTrainOptions {
  std::size_t m = 8;                        // Number of subspaces (bytes per code).
  std::size_t iterations = 25;              // Lloyd iterations per subspace.
  std::size_t max_training_points = 65536;  // Sampled without replacement.
  std::size_t threads = 1;
  std::uint64_t seed = 7;
};

class ProductQuantizer {
 public:
  ProductQuantizer() = default;

  // `data` holds `n` vectors, `stride` floats apart, of which the first `dim` are used.
  static ProductQuantizer train(const float* data, std::size_t n, std::size_t stride,
                                std::size_t dim, const PqTrainOptions& options);
  static ProductQuantizer from_codebook(std::size_t dim, std::size_t m,
                                        std::vector<float> codebook);

  void encode(const float* vector, std::uint8_t* code) const noexcept;
  void decode(const std::uint8_t* code, float* vector) const noexcept;

  // table[j * 256 + c] is the contribution of centroid c in subspace j. The sum over
  // subspaces approximates the squared L2 distance, or the dot product, to a code.
  void l2_table(const float* query, float* table) const noexcept;
  void dot_table(const float* query, float* table) const noexcept;

  std::size_t dim() const noexcept { return dim_; }
  std::size_t m() const noexcept { return m_; }
  std::span<const float> codebook() const noexcept { return codebook_; }
  std::size_t table_size() const noexcept { return m_ * kPqCentroids; }

 private:
  ProductQuantizer(std::size_t dim, std::size_t m);

  const float* centroid(std::size_t subspace, std::size_t index) const noexcept;

  std::size_t dim_ = 0;
  std::size_t m_ = 0;
  // Subspaces differ by at most one dimension when m does not divide dim, so
  // glove-25 with m=8 is not forced to drop or pad dimensions.
  std::vector<std::size_t> sub_offset_;
  std::vector<std::size_t> sub_dim_;
  std::vector<std::size_t> codebook_offset_;
  std::vector<float> codebook_;
};

}  // namespace annvil
