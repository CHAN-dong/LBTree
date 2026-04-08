cmake_minimum_required(VERSION 3.12)
project(LBTreeIndex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
  /home/dwang12576/.conda/envs/torch_env/include
  ${CMAKE_CURRENT_SOURCE_DIR}/index/include
  ${CMAKE_CURRENT_SOURCE_DIR}/index/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

add_executable(lbtree
  index/src/main.cpp
  index/src/LBTree.cpp
  index/src/Node.cpp
  index/src/WorkloadIO.cpp
  index/src/VocabIO.cpp
  index/src/BitmapIndexBaseline.cpp
)

# add_executable(costest
#   index/src/bitmap_sample_experiment.cpp
#   index/src/BitmapIndexBaseline.cpp
# )
