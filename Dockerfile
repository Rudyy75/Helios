# Helios - Multi-Stage Production Dockerfile
# Base: NVIDIA CUDA 11.8 + cuDNN 8 (Ubuntu 22.04)

# Stage 1: Build
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    ca-certificates \
    libasio-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Third-party dependencies (pre-compiled / header-only)

# ONNX Runtime GPU - pre-built binary (avoids 30-min source compilation)
ARG ORT_VERSION=1.18.0
RUN wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz \
    | tar xz && mv onnxruntime-linux-x64-gpu-${ORT_VERSION} /opt/onnxruntime

# Crow - single-header C++ HTTP framework
RUN mkdir -p /opt/crow && \
    wget -qO /opt/crow/crow.h \
    https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow_all.h

# nlohmann/json - single-header JSON library
RUN mkdir -p /opt/json/nlohmann && \
    
    wget -qO /opt/json/nlohmann/json.hpp \
    
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp

# Copy source
COPY include/ include/
COPY src/     src/
COPY benchmarks/ benchmarks/
COPY models/ models/

# Container-specific CMake (standalone, no vcpkg)
RUN printf 'cmake_minimum_required(VERSION 3.20)\n\
project(Helios LANGUAGES CXX)\n\
set(CMAKE_CXX_STANDARD 17)\n\
set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\
find_package(Threads REQUIRED)\n\
add_compile_definitions(USE_CUDA)\n\
add_library(inference_core src/inference_engine.cpp src/async_batcher.cpp)\n\
target_include_directories(inference_core PUBLIC include /opt/onnxruntime/include)\n\
target_link_libraries(inference_core PUBLIC /opt/onnxruntime/lib/libonnxruntime.so Threads::Threads)\n\
add_executable(cpp_benchmark benchmarks/cpp_benchmark.cpp)\ntarget_link_libraries(cpp_benchmark PRIVATE inference_core)\nadd_executable(helios_server src/server.cpp)\n\
target_include_directories(helios_server PRIVATE /opt/crow /opt/json)\n\
target_link_libraries(helios_server PRIVATE inference_core)\n' > CMakeLists.txt

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build build --config Release --target all -- -j$(nproc)

# Stage 2: Runtime (lean image - no compiler toolchain)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/onnxruntime/lib/libonnxruntime*.so* /usr/lib/
COPY --from=build /app/build/helios_server .
COPY --from=build /app/build/cpp_benchmark .
COPY --from=build /app/models/ models/

RUN ldconfig

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["./helios_server"]
CMD ["models/resnet50.onnx"]
