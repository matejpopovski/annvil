cmake_minimum_required(VERSION 3.24)
project(annvil VERSION 0.1.0 LANGUAGES CXX)

option(ANNVIL_BUILD_TESTS "Build the doctest suite" ON)
option(ANNVIL_BUILD_TOOLS "Build the annvil CLI and HTTP server" ON)
option(ANNVIL_BUILD_BENCH "Build the benchmark harness (fetches hnswlib)" OFF)
option(ANNVIL_BUILD_PYTHON "Build the pybind11 module" OFF)
option(ANNVIL_NATIVE "Tune for this CPU (-mcpu/-march=native)" OFF)
option(ANNVIL_FORCE_SCALAR "Compile out every SIMD kernel" OFF)
set(ANNVIL_SANITIZE "" CACHE STRING "Sanitizer list for -fsanitize, for example address,undefined")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ANNVIL_CPU_FLAGS "")
if(ANNVIL_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ANNVIL_CPU_FLAGS -mcpu=native)
  else()
    set(ANNVIL_CPU_FLAGS -march=native)
  endif()
endif()

if(ANNVIL_SANITIZE)
  add_compile_options(-fsanitize=${ANNVIL_SANITIZE} -fno-sanitize-recover=all
                      -fno-omit-frame-pointer -g -O1)
  add_link_options(-fsanitize=${ANNVIL_SANITIZE})
endif()

find_package(Threads REQUIRED)

add_library(annvil STATIC
  src/dataset.cpp
  src/index.cpp
  src/pq.cpp
  src/storage.cpp)
add_library(annvil::annvil ALIAS annvil)
target_include_directories(annvil
  PUBLIC include
  PRIVATE src)
target_link_libraries(annvil PUBLIC Threads::Threads)
target_compile_options(annvil PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
                                      -Wnon-virtual-dtor ${ANNVIL_CPU_FLAGS})
if(ANNVIL_FORCE_SCALAR)
  target_compile_definitions(annvil PUBLIC ANNVIL_FORCE_SCALAR)
endif()

include(FetchContent)

if(ANNVIL_BUILD_TOOLS)
  # Hash-pinned so a clean clone builds the same code that was tested.
  FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
    URL_HASH SHA256=a0567bcd6c3fe5cef1b329b96245119047f876b49e06cc129a36a7a8dffe173e)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  set(HTTPLIB_COMPILE OFF)
  set(HTTPLIB_REQUIRE_OPENSSL OFF)
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
  set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF)
  set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF)
  set(HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN OFF)
  set(JSON_BuildTests OFF)
  FetchContent_MakeAvailable(httplib nlohmann_json)

  add_library(annvil_server STATIC apps/server.cpp)
  target_include_directories(annvil_server PUBLIC apps)
  target_link_libraries(annvil_server
    PUBLIC annvil
    PRIVATE httplib::httplib nlohmann_json::nlohmann_json)
  target_compile_options(annvil_server PRIVATE -Wall -Wextra ${ANNVIL_CPU_FLAGS})

  add_executable(annvil_cli apps/annvil_cli.cpp)
  set_target_properties(annvil_cli PROPERTIES OUTPUT_NAME annvil)
  target_link_libraries(annvil_cli PRIVATE annvil annvil_server)
  target_compile_options(annvil_cli PRIVATE -Wall -Wextra ${ANNVIL_CPU_FLAGS})
endif()

if(ANNVIL_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
    URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186)
  FetchContent_GetProperties(doctest)
  if(NOT doctest_POPULATED)
    FetchContent_Populate(doctest)
  endif()

  set(ANNVIL_TEST_SOURCES
    tests/test_main.cpp
    tests/test_concurrency.cpp
    tests/test_dataset.cpp
    tests/test_hnsw.cpp
    tests/test_kernels.cpp
    tests/test_pq.cpp
    tests/test_storage.cpp)
  if(ANNVIL_BUILD_TOOLS)
    list(APPEND ANNVIL_TEST_SOURCES tests/test_server.cpp)
  endif()
  add_executable(annvil_tests ${ANNVIL_TEST_SOURCES})
  target_include_directories(annvil_tests PRIVATE tests src SYSTEM PRIVATE ${doctest_SOURCE_DIR})
  target_link_libraries(annvil_tests PRIVATE annvil)
  target_compile_options(annvil_tests PRIVATE -Wall -Wextra)
  if(ANNVIL_BUILD_TOOLS)
    target_link_libraries(annvil_tests PRIVATE annvil_server httplib::httplib)
    target_compile_definitions(annvil_tests PRIVATE ANNVIL_HAS_SERVER)
  endif()
  add_test(NAME annvil_tests COMMAND annvil_tests)
endif()

if(ANNVIL_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(ANNVIL_BUILD_PYTHON OR SKBUILD)
  add_subdirectory(python)
endif()
