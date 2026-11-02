// pybind11 bindings. Vectors cross as NumPy arrays without per-element conversion, and
// the GIL is released for everything that runs long, so Python threads searching
// one index really do run in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "annvil/index.hpp"
#include "annvil/kernels.hpp"
#include "annvil/parallel.hpp"

namespace py = pybind11;
using annvil::Index;
using annvil::Metric;
using annvil::Neighbor;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

Metric parse_metric(const std::string& name) {
  if (name == "l2") {
    return Metric::L2;
  }
  if (name == "angular") {
    return Metric::Angular;
  }
  throw std::invalid_argument("metric must be 'l2' or 'angular'");
}

std::string metric_name(Metric metric) {
  return metric == Metric::L2 ? "l2" : "angular";
}

struct Matrix {
  const float* data;
  std::size_t rows;
};

// Accepts one vector or a 2-D batch, and insists on the index's dimension.
Matrix as_matrix(const FloatArray& array, std::size_t dim) {
  if (array.ndim() == 1 && static_cast<std::size_t>(array.shape(0)) == dim) {
    return {array.data(), 1};
  }
  if (array.ndim() == 2 && static_cast<std::size_t>(array.shape(1)) == dim) {
    return {array.data(), static_cast<std::size_t>(array.shape(0))};
  }
  throw std::invalid_argument("expected vectors of dimension " + std::to_string(dim));
}

py::tuple batch_search(std::size_t rows, std::size_t k, std::size_t threads, const auto& one) {
  py::array_t<std::uint64_t> labels({rows, k});
  py::array_t<float> distances({rows, k});
  std::fill_n(labels.mutable_data(), rows * k, ~std::uint64_t{0});
  std::fill_n(distances.mutable_data(), rows * k, std::numeric_limits<float>::infinity());
  std::uint64_t* label_out = labels.mutable_data();
  float* distance_out = distances.mutable_data();
  {
    const py::gil_scoped_release release;
    annvil::parallel_for(
        rows, threads,
        [&](std::size_t row, std::size_t) {
          thread_local std::vector<Neighbor> found;
          found.resize(k);
          const std::size_t count = one(row, std::span<Neighbor>(found));
          for (std::size_t i = 0; i < count; ++i) {
            label_out[row * k + i] = found[i].label;
            distance_out[row * k + i] = found[i].distance;
          }
        },
        16);
  }
  return py::make_tuple(std::move(labels), std::move(distances));
}

}  // namespace

PYBIND11_MODULE(_annvil, m) {
  m.doc() = "HNSW vector search with mmap persistence and product quantization";
  m.attr("KERNEL") = annvil::kernels::active_kernel();

  py::class_<Index>(m, "Index")
      .def(py::init([](std::size_t dim, std::size_t capacity, const std::string& metric,
                       std::uint32_t M, std::uint32_t ef_construction, std::uint64_t seed) {
             annvil::IndexOptions options;
             options.dim = dim;
             options.capacity = capacity;
             options.metric = parse_metric(metric);
             options.M = M;
             options.ef_construction = ef_construction;
             options.seed = seed;
             return Index(options);
           }),
           py::arg("dim"), py::arg("capacity"), py::arg("metric") = "l2", py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 42)
      .def_static(
          "open", [](const std::string& path, bool verify) { return Index::open(path, verify); },
          py::arg("path"), py::arg("verify") = false,
          "Memory-map a saved index. The result is read-only.")
      .def("save", [](const Index& self, const std::string& path) { self.save(path); })
      .def(
          "add",
          [](Index& self, const FloatArray& vectors,
             const std::optional<py::array_t<std::uint64_t>>& labels, std::size_t threads) {
            const Matrix matrix = as_matrix(vectors, self.dim());
            std::span<const std::uint64_t> label_view;
            if (labels.has_value()) {
              label_view = {labels->data(), static_cast<std::size_t>(labels->size())};
            }
            const py::gil_scoped_release release;
            self.add_batch({matrix.data, matrix.rows * self.dim()}, threads, label_view);
          },
          py::arg("vectors"), py::arg("labels") = py::none(), py::arg("threads") = 1)
      .def(
          "search",
          [](const Index& self, const FloatArray& queries, std::size_t k, std::size_t ef,
             std::size_t threads) {
            const Matrix matrix = as_matrix(queries, self.dim());
            const std::size_t dim = self.dim();
            return batch_search(matrix.rows, k, threads,
                                [&](std::size_t row, std::span<Neighbor> out) {
                                  return self.search({matrix.data + row * dim, dim}, ef, out);
                                });
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 100, py::arg("threads") = 1,
          "Returns (labels, distances), each of shape (n_queries, k). Rows with fewer than k "
          "results are padded with the maximum uint64 label and infinite distance.")
      .def(
          "train_pq",
          [](Index& self, std::size_t m, std::size_t threads) {
            annvil::PqTrainOptions options;
            options.m = m;
            options.threads = threads;
            const py::gil_scoped_release release;
            self.attach_pq(options);
          },
          py::arg("m"), py::arg("threads") = 1)
      .def(
          "search_pq",
          [](const Index& self, const FloatArray& queries, std::size_t k, std::size_t ef,
             std::size_t rerank, std::size_t threads) {
            const Matrix matrix = as_matrix(queries, self.dim());
            const std::size_t dim = self.dim();
            return batch_search(
                matrix.rows, k, threads, [&](std::size_t row, std::span<Neighbor> out) {
                  return self.search_pq({matrix.data + row * dim, dim}, ef, rerank, out);
                });
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 100, py::arg("rerank") = 50,
          py::arg("threads") = 1)
      .def("reorder", &Index::reorder)
      .def("verify",
           [](const Index& self) {
             const annvil::VerifyReport report = self.verify();
             return py::make_tuple(report.ok, report.errors);
           })
      .def("__len__", &Index::size)
      .def_property_readonly("dim", &Index::dim)
      .def_property_readonly("capacity", &Index::capacity)
      .def_property_readonly("metric", [](const Index& self) { return metric_name(self.metric()); })
      .def_property_readonly("read_only", &Index::read_only)
      .def_property_readonly("has_pq", &Index::has_pq)
      .def_property_readonly("memory", [](const Index& self) {
        const annvil::MemoryBreakdown memory = self.memory();
        py::dict out;
        out["vectors"] = memory.vectors;
        out["links"] = memory.links;
        out["labels"] = memory.labels;
        out["pq"] = memory.pq;
        return out;
      });
}
