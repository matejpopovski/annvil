# Benchmark harness. hnswlib is pinned and built from source here, in two forms:
#
#   hnswlib        v0.8.0 unmodified. On ARM its SIMD and its prefetches are compiled
#                  out (both are guarded by USE_SSE), so it relies on auto-vectorization.
#                  It gets -O3 -ffast-math and CPU tuning, as its own CMakeLists asks for.
#   hnswlib-tuned  The same source with three changes, all favouring hnswlib: prefetches
#                  are mapped to __builtin_prefetch, the ones aimed at a vector cover
#                  the whole vector rather than its first line, and distances use
#                  annvil's NEON kernels through its public SpaceInterface.
include(FetchContent)
FetchContent_Declare(hnswlib_src
  URL https://github.com/nmslib/hnswlib/archive/refs/tags/v0.8.0.tar.gz
  URL_HASH SHA256=cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb)
FetchContent_GetProperties(hnswlib_src)
if(NOT hnswlib_src_POPULATED)
  FetchContent_Populate(hnswlib_src)
endif()

set(HNSWLIB_FLAGS -O3 -ffast-math ${ANNVIL_CPU_FLAGS})

add_library(engine_hnswlib OBJECT engine_hnswlib.cpp)
target_include_directories(engine_hnswlib PRIVATE ${CMAKE_SOURCE_DIR}/include
                           SYSTEM PRIVATE ${hnswlib_src_SOURCE_DIR})
target_compile_options(engine_hnswlib PRIVATE ${HNSWLIB_FLAGS})
target_compile_definitions(engine_hnswlib PRIVATE ANNVIL_ENGINE_FACTORY=make_hnswlib_engine)

# Derive the tuned copy by editing the source at configure time, so the exact delta
# from upstream is these lines and nothing else.
set(TUNED_DIR ${CMAKE_CURRENT_BINARY_DIR}/hnswlib_tuned/hnswlib)
file(MAKE_DIRECTORY ${TUNED_DIR})
file(GLOB HNSWLIB_HEADERS ${hnswlib_src_SOURCE_DIR}/hnswlib/*.h)
file(COPY ${HNSWLIB_HEADERS} DESTINATION ${TUNED_DIR})
file(READ ${hnswlib_src_SOURCE_DIR}/hnswlib/hnswalg.h HNSWALG)
string(REPLACE "#ifdef USE_SSE" "#if 1" HNSWALG "${HNSWALG}")
string(REGEX REPLACE
  "_mm_prefetch\\(data_level0_memory_ \\+ \\(\\*\\(data \\+ ([^;]*)\\)\\) \\* size_data_per_element_ \\+ offsetData_,[ \n]*_MM_HINT_T0\\);"
  "ANNVIL_PREFETCH_VECTOR(data_level0_memory_ + (*(data + \\1)) * size_data_per_element_ + offsetData_);"
  HNSWALG "${HNSWALG}")
string(REGEX REPLACE
  "_mm_prefetch\\(getDataByInternalId\\(([^;]*)\\), _MM_HINT_T0\\);"
  "ANNVIL_PREFETCH_VECTOR(getDataByInternalId(\\1));"
  HNSWALG "${HNSWALG}")
string(PREPEND HNSWALG [=[
#ifndef _mm_prefetch
#define _mm_prefetch(p, hint) __builtin_prefetch(p)
#endif
#if defined(__APPLE__) && defined(__aarch64__)
#define ANNVIL_LINE_BYTES 128
#else
#define ANNVIL_LINE_BYTES 64
#endif
#define ANNVIL_PREFETCH_VECTOR(p)                                    \
  do {                                                               \
    const char* start_ = reinterpret_cast<const char*>(p);           \
    for (size_t at_ = 0; at_ < data_size_; at_ += ANNVIL_LINE_BYTES) \
      __builtin_prefetch(start_ + at_);                              \
  } while (0)
]=])
file(WRITE ${TUNED_DIR}/hnswalg.h "${HNSWALG}")

add_library(engine_hnswlib_tuned OBJECT engine_hnswlib.cpp)
target_include_directories(engine_hnswlib_tuned PRIVATE ${CMAKE_SOURCE_DIR}/include
                           SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/hnswlib_tuned)
target_compile_options(engine_hnswlib_tuned PRIVATE ${HNSWLIB_FLAGS})
# Renaming the namespace keeps the two copies of the templates apart at link time.
target_compile_definitions(engine_hnswlib_tuned PRIVATE
  ANNVIL_HNSWLIB_TUNED ANNVIL_ENGINE_FACTORY=make_hnswlib_tuned_engine hnswlib=hnswlib_tuned)

add_executable(annvil_bench annvil_bench.cpp
  $<TARGET_OBJECTS:engine_hnswlib>
  $<TARGET_OBJECTS:engine_hnswlib_tuned>)
target_link_libraries(annvil_bench PRIVATE annvil)
target_compile_options(annvil_bench PRIVATE -Wall -Wextra ${ANNVIL_CPU_FLAGS})
target_compile_definitions(annvil_bench PRIVATE
  "ANNVIL_BENCH_ANNVIL_FLAGS=\"-O3 ${ANNVIL_CPU_FLAGS}\""
  "ANNVIL_BENCH_HNSWLIB_FLAGS=\"-O3 -ffast-math ${ANNVIL_CPU_FLAGS}\"")

# Parallel build quality study. The top-down variant is the shipped src/index.cpp with
# bench/top_down_linking.patch applied, so the core carries no experiment code, and a
# patch that stops applying fails the build rather than silently measuring nothing.
find_program(PATCH_PROGRAM patch REQUIRED)
set(TOP_DOWN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/index_top_down.cpp)
add_custom_command(OUTPUT ${TOP_DOWN_SOURCE}
  COMMAND ${PATCH_PROGRAM} --silent -o ${TOP_DOWN_SOURCE}
          ${CMAKE_SOURCE_DIR}/src/index.cpp ${CMAKE_CURRENT_SOURCE_DIR}/top_down_linking.patch
  DEPENDS ${CMAKE_SOURCE_DIR}/src/index.cpp ${CMAKE_CURRENT_SOURCE_DIR}/top_down_linking.patch
  VERBATIM)

add_library(annvil_top_down STATIC ${TOP_DOWN_SOURCE} ${CMAKE_SOURCE_DIR}/src/dataset.cpp
            ${CMAKE_SOURCE_DIR}/src/pq.cpp ${CMAKE_SOURCE_DIR}/src/storage.cpp)
target_include_directories(annvil_top_down PUBLIC ${CMAKE_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(annvil_top_down PUBLIC Threads::Threads)
target_compile_options(annvil_top_down PRIVATE ${ANNVIL_CPU_FLAGS})

# Same idea for the prefetch ablation: annvil with hnswlib's search loop, to measure
# what prefetching every neighbour before the first distance is worth.
set(ONE_AHEAD_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/index_one_ahead.cpp)
add_custom_command(OUTPUT ${ONE_AHEAD_SOURCE}
  COMMAND ${PATCH_PROGRAM} --silent -o ${ONE_AHEAD_SOURCE}
          ${CMAKE_SOURCE_DIR}/src/index.cpp ${CMAKE_CURRENT_SOURCE_DIR}/one_ahead_prefetch.patch
  DEPENDS ${CMAKE_SOURCE_DIR}/src/index.cpp ${CMAKE_CURRENT_SOURCE_DIR}/one_ahead_prefetch.patch
  VERBATIM)

add_library(annvil_one_ahead STATIC ${ONE_AHEAD_SOURCE} ${CMAKE_SOURCE_DIR}/src/dataset.cpp
            ${CMAKE_SOURCE_DIR}/src/pq.cpp ${CMAKE_SOURCE_DIR}/src/storage.cpp)
target_include_directories(annvil_one_ahead PUBLIC ${CMAKE_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(annvil_one_ahead PUBLIC Threads::Threads)
target_compile_options(annvil_one_ahead PRIVATE ${ANNVIL_CPU_FLAGS})

add_executable(annvil_bench_one_ahead annvil_bench.cpp
  $<TARGET_OBJECTS:engine_hnswlib>
  $<TARGET_OBJECTS:engine_hnswlib_tuned>)
target_link_libraries(annvil_bench_one_ahead PRIVATE annvil_one_ahead)
target_compile_options(annvil_bench_one_ahead PRIVATE -Wall -Wextra ${ANNVIL_CPU_FLAGS})
target_compile_definitions(annvil_bench_one_ahead PRIVATE
  "ANNVIL_BENCH_ANNVIL_FLAGS=\"-O3 ${ANNVIL_CPU_FLAGS}\""
  "ANNVIL_BENCH_HNSWLIB_FLAGS=\"-O3 -ffast-math ${ANNVIL_CPU_FLAGS}\"")

foreach(variant annvil annvil_top_down)
  add_executable(${variant}_parallel_quality parallel_quality.cpp)
  target_include_directories(${variant}_parallel_quality PRIVATE ${CMAKE_SOURCE_DIR}/tests
                             SYSTEM PRIVATE ${hnswlib_src_SOURCE_DIR})
  target_link_libraries(${variant}_parallel_quality PRIVATE ${variant})
  target_compile_options(${variant}_parallel_quality PRIVATE ${HNSWLIB_FLAGS})
endforeach()
