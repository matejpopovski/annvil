#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace annvil {

// Runs fn(i, worker) for i in [0, n) on `threads` workers, handing out indices in
// small chunks. The harness drives annvil and hnswlib through this same function
// so that a build-time comparison measures the engines, not the schedulers.
// The first exception cancels remaining work and is rethrown on the caller.
template <class Fn>
void parallel_for(std::size_t n, std::size_t threads, Fn&& fn, std::size_t chunk = 64) {
  if (threads <= 1 || n <= chunk) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i, std::size_t{0});
    }
    return;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr first_error;
  std::mutex error_mutex;

  auto worker = [&](std::size_t worker_id) {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t begin = next.fetch_add(chunk, std::memory_order_relaxed);
      if (begin >= n) {
        return;
      }
      const std::size_t end = begin + chunk < n ? begin + chunk : n;
      try {
        for (std::size_t i = begin; i < end; ++i) {
          fn(i, worker_id);
        }
      } catch (...) {
        const std::scoped_lock lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
        failed.store(true, std::memory_order_relaxed);
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (std::size_t t = 1; t < threads; ++t) {
    pool.emplace_back(worker, t);
  }
  worker(0);
  for (auto& thread : pool) {
    thread.join();
  }
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

}  // namespace annvil
