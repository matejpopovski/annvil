#include <doctest/doctest.h>
#include <httplib.h>

#include <string>
#include <thread>
#include <vector>

#include "annvil/index.hpp"
#include "server.hpp"
#include "test_util.hpp"

using namespace annvil;
using namespace annvil::test;

namespace {

constexpr std::size_t kDim = 4;

Index tiny_index() {
  const auto data = clustered(300, kDim, 51, 8);
  IndexOptions o;
  o.dim = kDim;
  o.capacity = 300;
  Index index(o);
  index.add_batch(data, 1);
  return index;
}

std::string vector_json(const float* v) {
  std::string out = "[";
  for (std::size_t i = 0; i < kDim; ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(v[i]);
  }
  return out + "]";
}

}  // namespace

TEST_CASE("the search handler validates input before touching the index") {
  const Index index = tiny_index();
  const ServerOptions options;
  const auto status = [&](const std::string& body) {
    return handle_search(index, options, body).status;
  };
  CHECK(status("not json") == 400);
  CHECK(status("[1,2,3]") == 400);
  CHECK(status(R"({"k":3})") == 400);
  CHECK(status(R"({"vector":"nope"})") == 400);
  CHECK(status(R"({"vector":[1,2]})") == 400);
  CHECK(status(R"({"vector":[1,2,3,"x"]})") == 400);
  CHECK(status(R"({"vector":[1,2,3,1e999]})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"k":0})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"k":-1})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"k":"ten"})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"k":100000})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"ef":100000000})") == 400);
  CHECK(status(R"({"vector":[1,2,3,4],"k":5,"ef":50})") == 200);
}

TEST_CASE("the server answers real HTTP requests") {
  TempDir dir;
  const Index built = tiny_index();
  built.save(dir.file("s.annvil"));
  const Index mapped = Index::open(dir.file("s.annvil"));

  ServerOptions options;
  options.port = 0;
  options.threads = 2;
  Server server(mapped, options);
  const int port = server.bind();
  REQUIRE(port > 0);
  std::thread loop([&] { server.run(); });

  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(5);

  const auto health = client.Get("/health");
  REQUIRE(health);
  CHECK(health->status == 200);

  const auto info = client.Get("/v1/info");
  REQUIRE(info);
  CHECK(info->status == 200);
  CHECK(info->body.find("\"size\":300") != std::string::npos);

  const auto data = clustered(300, kDim, 51, 8);
  const std::string body = R"({"vector":)" + vector_json(data.data()) + R"(,"k":3})";
  const auto found = client.Post("/v1/search", body, "application/json");
  REQUIRE(found);
  CHECK(found->status == 200);
  CHECK(found->body.find("\"label\":0") != std::string::npos);

  const auto rejected = client.Post("/v1/search", "{}", "application/json");
  REQUIRE(rejected);
  CHECK(rejected->status == 400);
  CHECK(rejected->body.find("error") != std::string::npos);

  const auto missing = client.Get("/nope");
  REQUIRE(missing);
  CHECK(missing->status == 404);

  server.stop();
  loop.join();
}

TEST_CASE("oversized request bodies are refused") {
  const Index index = tiny_index();
  ServerOptions options;
  options.port = 0;
  options.max_body_bytes = 256;
  Server server(index, options);
  const int port = server.bind();
  std::thread loop([&] { server.run(); });

  httplib::Client client("127.0.0.1", port);
  const auto response = client.Post("/v1/search", std::string(4096, ' '), "application/json");
  REQUIRE(response);
  CHECK(response->status == 413);

  server.stop();
  loop.join();
}
