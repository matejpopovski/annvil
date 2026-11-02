// Distance kernels. The right one is chosen at compile time so the search loop
// can inline it. NEON is the tuned path. SSE2 and AVX2 exist so x86 builds stay
// correct and reasonable, and the scalar kernels are the reference the tests
// compare against.
#pragma once

#include <cstddef>

#if !defined(ANNVIL_FORCE_SCALAR)
#if defined(__ARM_NEON) || defined(__aarch64__)
#define ANNVIL_HAVE_NEON 1
#include <arm_neon.h>
#endif
#if defined(__SSE2__) || defined(__x86_64__)
#define ANNVIL_HAVE_SSE2 1
#include <immintrin.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#define ANNVIL_HAVE_AVX2 1
#endif
#endif

namespace annvil::kernels {

inline float l2_sq_scalar(const float* a, const float* b, std::size_t n) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

inline float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

#if defined(ANNVIL_HAVE_NEON)
// Four accumulators, so the FMLAs in one iteration are independent. A single
// accumulator would serialize them on its own register dependency.
inline float l2_sq_neon(const float* a, const float* b, std::size_t n) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = s0, s2 = s0, s3 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= n; i += 4) {
    const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, d, d);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

inline float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = s0, s2 = s0, s3 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
#endif

#if defined(ANNVIL_HAVE_SSE2)
inline float hsum_sse(__m128 v) noexcept {
  const __m128 shuffled = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128 sums = _mm_add_ps(v, shuffled);
  const __m128 high = _mm_movehl_ps(shuffled, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, high));
}

inline float l2_sq_sse(const float* a, const float* b, std::size_t n) noexcept {
  __m128 acc = _mm_setzero_ps();
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
    acc = _mm_add_ps(acc, _mm_mul_ps(d, d));
  }
  float sum = hsum_sse(acc);
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

inline float dot_sse(const float* a, const float* b, std::size_t n) noexcept {
  __m128 acc = _mm_setzero_ps();
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
  }
  float sum = hsum_sse(acc);
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
#endif

#if defined(ANNVIL_HAVE_AVX2)
inline float hsum_avx(__m256 v) noexcept {
  return hsum_sse(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)));
}

inline float l2_sq_avx2(const float* a, const float* b, std::size_t n) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc = _mm256_fmadd_ps(d, d, acc);
  }
  float sum = hsum_avx(acc);
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

inline float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  }
  float sum = hsum_avx(acc);
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
#endif

inline float l2_sq(const float* a, const float* b, std::size_t n) noexcept {
#if defined(ANNVIL_HAVE_NEON)
  return l2_sq_neon(a, b, n);
#elif defined(ANNVIL_HAVE_AVX2)
  return l2_sq_avx2(a, b, n);
#elif defined(ANNVIL_HAVE_SSE2)
  return l2_sq_sse(a, b, n);
#else
  return l2_sq_scalar(a, b, n);
#endif
}

inline float dot(const float* a, const float* b, std::size_t n) noexcept {
#if defined(ANNVIL_HAVE_NEON)
  return dot_neon(a, b, n);
#elif defined(ANNVIL_HAVE_AVX2)
  return dot_avx2(a, b, n);
#elif defined(ANNVIL_HAVE_SSE2)
  return dot_sse(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

// Reported by `annvil info` and the benchmark metadata so results always say
// which code path produced them.
inline const char* active_kernel() noexcept {
#if defined(ANNVIL_HAVE_NEON)
  return "neon";
#elif defined(ANNVIL_HAVE_AVX2)
  return "avx2";
#elif defined(ANNVIL_HAVE_SSE2)
  return "sse2";
#else
  return "scalar";
#endif
}

}  // namespace annvil::kernels
