cmake_minimum_required(VERSION 3.14)
project(LBTreeIndex CXX)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(lbtree_stream SHARED index/src/StreamingAssignNative.cpp)
if(MSVC)
  target_compile_options(lbtree_stream PRIVATE /O2 /fp:strict)
else()
  target_compile_options(lbtree_stream PRIVATE -O3 -ffp-contract=off -fno-fast-math)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/index/include
  ${CMAKE_CURRENT_SOURCE_DIR}/index/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

add_executable(lbtree_small_bench
  benchmarks/small.cpp index/src/LBTree.cpp index/src/Node.cpp)
enable_testing()
add_test(NAME small_correctness COMMAND lbtree_small_bench 4096)

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  find_path(LBTREE_JSON_INCLUDE nlohmann/json.hpp
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/build-local/json-include)
  if(LBTREE_JSON_INCLUDE)
    add_library(lbtree_json INTERFACE)
    target_include_directories(lbtree_json INTERFACE ${LBTREE_JSON_INCLUDE})
    add_library(nlohmann_json::nlohmann_json ALIAS lbtree_json)
  else()
  include(FetchContent)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
  endif()
endif()

add_executable(lbtree
  index/src/main.cpp
  index/src/LBTree.cpp
  index/src/Node.cpp
  index/src/WorkloadIO.cpp
  index/src/VocabIO.cpp
  index/src/BitmapIndexBaseline.cpp
)
target_link_libraries(lbtree PRIVATE nlohmann_json::nlohmann_json)

add_executable(lbtree_workload_bench
  benchmarks/workload_bench.cpp index/src/LBTree.cpp index/src/Node.cpp
  index/src/WorkloadIO.cpp index/src/VocabIO.cpp index/src/BitmapIndexBaseline.cpp)
target_link_libraries(lbtree_workload_bench PRIVATE nlohmann_json::nlohmann_json)

# add_executable(costest
#   index/src/bitmap_sample_experiment.cpp
#   index/src/BitmapIndexBaseline.cpp
# )
