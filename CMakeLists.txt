cmake_minimum_required(VERSION 3.20)
project(Helios LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

find_package(onnxruntime CONFIG REQUIRED)
find_package(Crow CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)

option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
  add_compile_definitions(USE_CUDA)
endif()

add_library(inference_core
    src/inference_engine.cpp
    src/async_batcher.cpp
)
target_include_directories(inference_core PUBLIC include)
target_link_libraries(inference_core PUBLIC onnxruntime::onnxruntime Threads::Threads)

add_executable(helios_server src/server.cpp)
target_link_libraries(helios_server PRIVATE
    inference_core
    Crow::Crow
    nlohmann_json::nlohmann_json
)

# Unit tests (optional target)
enable_testing()
find_package(GTest CONFIG REQUIRED)
add_executable(unit_tests
    tests/test_inference_engine.cpp
    tests/test_async_batcher.cpp
)
target_link_libraries(unit_tests PRIVATE inference_core GTest::gtest GTest::gtest_main)
add_test(NAME unit_tests COMMAND unit_tests)

add_executable(cpp_benchmark benchmarks/cpp_benchmark.cpp)
target_link_libraries(cpp_benchmark PRIVATE inference_core)

