// Read-only search over HTTP. HTTP parsing and JSON are delegated to cpp-httplib and
// nlohmann/json so this file holds only the request contract.
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "annvil/index.hpp"

namespace annvil {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;  // 0 picks a free port; bind() returns it.
  std::size_t threads = 4;
  std::size_t default_k = 10;
  std::size_t default_ef = 100;
  std::size_t max_k = 1000;
  std::size_t max_ef = 16384;
  std::size_t max_body_bytes = 1 << 20;
};

struct HandlerResponse {
  int status = 200;
  std::string body;
};

// Handles POST /v1/search. Kept separate from the socket layer so its validation
// can be tested without a network. Body: {"vector": [...], "k": 10, "ef": 100}.
HandlerResponse handle_search(const Index& index, const ServerOptions& options,
                              const std::string& body);

class Server {
 public:
  Server(const Index& index, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  int bind();  // Returns the bound port.
  void run();  // Blocks until stop().
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace annvil
