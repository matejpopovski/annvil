#include <doctest/doctest.h>

#include <random>
#include <vector>

#include "annvil/kernels.hpp"

using namespace annvil::kernels;

namespace {

struct Pair {
  std::vector<float> a, b;
};

// Offset one float past a cache line and page boundaries the kernels never assume.
Pair random_pair(std::size_t n, std::mt19937& rng, std::size_t misalign) {
  std::uniform_real_distribution<float> value(-10.0f, 10.0f);
  Pair p{std::vector<float>(n + misalign), std::vector<float>(n + misalign)};
  for (std::size_t i = misalign; i < n + misalign; ++i) {
    p.a[i] = value(rng);
    p.b[i] = value(rng);
  }
  return p;
}

// Compared in double because SIMD changes the summation order, so exact float
// equality with the scalar loop is not a property worth asserting.
double reference_l2(const float* a, const float* b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    sum += d * d;
  }
  return sum;
}

double reference_dot(const float* a, const float* b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return sum;
}

template <class L2, class Dot>
void check_kernel(L2 l2, Dot dot) {
  std::mt19937 rng(1234);
  for (std::size_t n = 0; n <= 300; ++n) {
    for (std::size_t misalign : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
      const Pair p = random_pair(n, rng, misalign);
      const float* a = p.a.data() + misalign;
      const float* b = p.b.data() + misalign;
      const double expected_l2 = reference_l2(a, b, n);
      const double expected_dot = reference_dot(a, b, n);
      CHECK(static_cast<double>(l2(a, b, n)) ==
            doctest::Approx(expected_l2).epsilon(1e-4).scale(1.0));
      CHECK(static_cast<double>(dot(a, b, n)) ==
            doctest::Approx(expected_dot).epsilon(1e-3).scale(50.0));
    }
  }
}

}  // namespace

TEST_CASE("scalar kernels match the double-precision reference") {
  check_kernel(l2_sq_scalar, dot_scalar);
}

#if defined(ANNVIL_HAVE_NEON)
TEST_CASE("NEON kernels match the reference at every length and alignment") {
  check_kernel(l2_sq_neon, dot_neon);
}
#endif

#if defined(ANNVIL_HAVE_SSE2)
TEST_CASE("SSE2 kernels match the reference at every length and alignment") {
  check_kernel(l2_sq_sse, dot_sse);
}
#endif

#if defined(ANNVIL_HAVE_AVX2)
TEST_CASE("AVX2 kernels match the reference at every length and alignment") {
  check_kernel(l2_sq_avx2, dot_avx2);
}
#endif

TEST_CASE("the dispatched kernel names a real code path") {
  const std::string name = active_kernel();
  CHECK((name == "neon" || name == "sse2" || name == "avx2" || name == "scalar"));
  std::mt19937 rng(9);
  const Pair p = random_pair(64, rng, 0);
  CHECK(static_cast<double>(l2_sq(p.a.data(), p.b.data(), 64)) ==
        doctest::Approx(reference_l2(p.a.data(), p.b.data(), 64)).epsilon(1e-4));
}

TEST_CASE("kernels read nothing past the requested length") {
  // Sentinels after the end would change the result if a loop overran.
  std::vector<float> a(19, 1.0f), b(19, 2.0f);
  a[17] = 1e6f;
  a[18] = 1e6f;
  CHECK(l2_sq(a.data(), b.data(), 17) == doctest::Approx(17.0f));
  CHECK(dot(a.data(), b.data(), 17) == doctest::Approx(34.0f));
}
