cmake_minimum_required(VERSION 3.22.1)
project(neuralprobe LANGUAGES CXX)

add_library(neuralprobe SHARED neuralprobe.cpp espcn_bench.cpp)
set_target_properties(neuralprobe PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(neuralprobe neuralnetworks log android)
