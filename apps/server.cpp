#include "server.hpp"

#include <httplib.h>

#include <atomic>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "annvil/kernels.hpp"

namespace annvil {
namespace {

using nlohmann::json;

HandlerResponse error(int status, const std::string& message) {
  return {status, json{{"error", message}}.dump()};
}

// Reads an optional positive integer field. Returns false and fills `problem` when
// the field is present but unusable.
bool read_limit(const json& request, const char* field, std::size_t maximum, std::size_t& value,
                std::string& problem) {
  const auto it = request.find(field);
  if (it == request.end()) {
    return true;
  }
  if (!it->is_number_integer() || it->get<std::int64_t>() < 1) {
    problem = std::string(field) + " must be a positive integer";
    return false;
  }
  const auto parsed = it->get<std::uint64_t>();
  if (parsed > maximum) {
    problem = std::string(field) + " must be at most " + std::to_string(maximum);
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

}  // namespace

HandlerResponse handle_search(const Index& index, const ServerOptions& options,
                              const std::string& body) {
  const json request = json::parse(body, nullptr, false);
  if (request.is_discarded() || !request.is_object()) {
    return error(400, "body must be a JSON object");
  }
  const auto vector_field = request.find("vector");
  if (vector_field == request.end() || !vector_field->is_array()) {
    return error(400, "vector must be an array of numbers");
  }
  if (vector_field->size() != index.dim()) {
    return error(400, "vector must have " + std::to_string(index.dim()) + " elements");
  }
  std::vector<float> query;
  query.reserve(index.dim());
  for (const json& element : *vector_field) {
    if (!element.is_number()) {
      return error(400, "vector must be an array of numbers");
    }
    const auto value = element.get<float>();
    if (!std::isfinite(value)) {
      return error(400, "vector must contain finite numbers");
    }
    query.push_back(value);
  }

  std::size_t k = options.default_k;
  std::size_t ef = options.default_ef;
  std::string problem;
  if (!read_limit(request, "k", options.max_k, k, problem) ||
      !read_limit(request, "ef", options.max_ef, ef, problem)) {
    return error(400, problem);
  }

  std::vector<Neighbor> found(k);
  const std::size_t count = index.search(query, ef, found);
  json results = json::array();
  for (std::size_t i = 0; i < count; ++i) {
    results.push_back({{"label", found[i].label}, {"distance", found[i].distance}});
  }
  return {200, json{{"results", std::move(results)}}.dump()};
}

struct Server::Impl {
  Impl(const Index& ix, ServerOptions opts) : index(ix), options(std::move(opts)) {}

  const Index& index;
  ServerOptions options;
  httplib::Server http;
  std::atomic<std::uint64_t> searches{0};
};

Server::Server(const Index& index, ServerOptions options)
    : impl_(std::make_unique<Impl>(index, std::move(options))) {
  Impl& s = *impl_;
  s.http.new_task_queue = [threads = s.options.threads] {
    return new httplib::ThreadPool(threads);
  };
  s.http.set_payload_max_length(s.options.max_body_bytes);

  s.http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });
  s.http.Get("/v1/info", [&s](const httplib::Request&, httplib::Response& res) {
    const json info = {{"size", s.index.size()},
                       {"dim", s.index.dim()},
                       {"metric", s.index.metric() == Metric::L2 ? "l2" : "angular"},
                       {"M", s.index.M()},
                       {"kernel", kernels::active_kernel()},
                       {"searches", s.searches.load(std::memory_order_relaxed)}};
    res.set_content(info.dump(), "application/json");
  });
  s.http.Post("/v1/search", [&s](const httplib::Request& req, httplib::Response& res) {
    const HandlerResponse out = handle_search(s.index, s.options, req.body);
    res.status = out.status;
    res.set_content(out.body, "application/json");
    s.searches.fetch_add(1, std::memory_order_relaxed);
  });
}

Server::~Server() = default;

int Server::bind() {
  Impl& s = *impl_;
  if (s.options.port == 0) {
    const int port = s.http.bind_to_any_port(s.options.host);
    if (port < 0) {
      throw std::runtime_error("cannot bind to " + s.options.host);
    }
    s.options.port = port;
    return port;
  }
  if (!s.http.bind_to_port(s.options.host, s.options.port)) {
    throw std::runtime_error("cannot bind to " + s.options.host + ":" +
                             std::to_string(s.options.port));
  }
  return s.options.port;
}

void Server::run() {
  impl_->http.listen_after_bind();
}

void Server::stop() {
  impl_->http.wait_until_ready();
  impl_->http.stop();
}

}  // namespace annvil
