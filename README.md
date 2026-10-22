<div align="center">

# ⚡ Helios

**High-performance C++ model inference server built on ONNX Runtime**

*Dynamic batching · CUDA GPU acceleration · REST API · Docker-ready*

*Eliminating Python runtime & GIL overhead for ~1.8x lower CPU latency · Sub-4ms latency & 515+ img/s with CUDA GPU acceleration*

---

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18-purple?logo=onnx)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

## Overview

Helios is a production-style C++ inference server that takes a pretrained model (exported to ONNX), serves predictions over a REST API, and automatically batches concurrent requests for maximum throughput. It demonstrates the kind of ML infrastructure engineering used at companies like NVIDIA (Triton), AWS (SageMaker), and Meta (TorchServe) — but built entirely from scratch.

### Why not just use PyTorch + FastAPI?

Standard Python inference endpoints incur significant overhead from the Global Interpreter Lock (GIL), serialization, and Python runtime layers. Helios delivers two distinct layers of performance:

1. **Like-for-like CPU speedup (~1.8x faster):** On identical CPU hardware (4 threads), pure C++ execution with ONNX Runtime graph optimizations reduces mean latency from **42.48 ms** (PyTorch CPU) to **23.82 ms** (Helios CPU).
2. **GPU hardware acceleration (~12.4x throughput):** Offloading execution to CUDA drops latency to **3.91 ms** and achieves **515.5 img/s** throughput under dynamic batching.

| Mode | Hardware | Mean Latency (batch=1) | p95 Latency | Throughput (batch=4) | Comparison |
|------|----------|------------------------|-------------|----------------------|------------|
| **PyTorch CPU** (Python) | CPU (4 threads) | 42.48 ms | 48.06 ms | N/A | Baseline |
| **Helios C++ Core** | CPU (4 threads) | 23.82 ms | 25.72 ms | 41.63 img/s | **~1.8x faster** (apples-to-apples CPU) |
| **Helios C++ Core** | GPU (CUDA 11.8) | **3.91 ms** | **4.45 ms** | **515.48 img/s** | **~6.1x faster than C++ CPU** |

> *Note on comparisons: CPU benchmarks were run on Intel/AMD x86_64 matching thread counts (4 threads). GPU benchmarks were run inside the NVIDIA CUDA 11.8 Docker container. While Helios GPU is ~10.8x faster than the PyTorch CPU baseline, that is a cross-hardware comparison (GPU vs. CPU); the fair architectural software comparison on identical hardware is the CPU-to-CPU ~1.8x speedup.*

## Architecture

```
         Client (HTTP/JSON)
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Crow REST API                               │
│  POST /infer  ·  POST /infer/batch           │
│  GET  /health ·  GET  /metrics               │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  AsyncBatcher                                │
│  Thread-safe queue · Promise/Future          │
│  Deadline-based batch formation (5ms window) │
│  Configurable max batch size (default: 8)    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  InferenceEngine                             │
│  ONNX Runtime C++ Session                    │
│  Memory arena · Graph optimization (ALL)     │
│  CUDA EP (auto-fallback to CPU)              │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
            ResNet-50 (ONNX)
           Dynamic batch axis
```

## Key Engineering Decisions

- **True batched inference** — Concurrent REST requests are stacked into a single tensor and executed in one `Session::Run()` call, not looped individually. This is the #1 optimization that production inference servers use.
- **Deadline-based batch formation** — The batcher uses `condition_variable::wait_until` with a configurable timeout window (default 5ms), balancing latency vs. throughput without polling.
- **CUDA with graceful degradation** — When compiled with `-DUSE_CUDA=ON`, the engine attempts to bind to the NVIDIA CUDA Execution Provider. If no GPU is available, it silently falls back to CPU.
- **String lifetime safety** — ONNX Runtime's C++ API returns allocated name pointers that must outlive the session. We store owned `std::string` copies to prevent dangling pointer UB (a common bug in naive ORT integrations).
- **Multi-stage Docker build** — The container compiles with the full CUDA dev toolkit, then copies only the stripped binary + shared libs into a lean runtime image.

## Project Structure

```
Helios/
├── include/
│   ├── inference_engine.hpp    # ONNX Runtime session wrapper
│   └── async_batcher.hpp       # Thread-safe dynamic batcher & server metrics
├── src/
│   ├── inference_engine.cpp    # Core inference + CUDA EP logic
│   ├── async_batcher.cpp       # Batch queue + worker thread
│   └── server.cpp              # Crow REST API + routing
├── benchmarks/
│   ├── cpp_benchmark.cpp       # Raw C++ latency/throughput benchmark
│   ├── benchmark.py            # PyTorch baseline + server comparison
│   └── results/
│       └── benchmark_results.md# Documented benchmark run logs
├── tests/
│   ├── test_inference_engine.cpp
│   ├── test_async_batcher.cpp
│   └── smoke_test.cpp
├── models/
│   └── export_model.py         # PyTorch → ONNX export script
├── scripts/
│   ├── setup_env.ps1           # Windows dependency setup
│   ├── setup_env.sh            # Linux dependency setup
│   └── run_server.sh           # Server launch helper script
├── CMakeLists.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## Quick Start

### Option 1: Local Build (Windows / Linux)

**Prerequisites:** CMake 3.20+, C++17 compiler (MSVC / GCC / Clang), Python 3.10+

```bash
# 1. Clone and set up dependencies
git clone https://github.com/yourusername/Helios.git && cd Helios
./scripts/setup_env.sh        # or .\scripts\setup_env.ps1 on Windows

# 2. Export the model (downloads pretrained ResNet-50 → ONNX)
python models/export_model.py

# 3. Build
cmake -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# 4. Run the server
./build/Release/helios_server models/resnet50.onnx
```

### Option 2: Docker (GPU)

**Prerequisites:** Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
docker-compose up --build
```

The server starts on port `8080` with automatic GPU detection.

## API Reference

### `POST /infer` — Single Inference

```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"data": [0.5, 0.5, ...], "shape": [3, 224, 224]}'
```

**Response:**
```json
{
  "output": [0.12, 0.03, ...],
  "shape": [1000],
  "latency_ms": 24.5
}
```

### `POST /infer/batch` — Batch Inference

```bash
curl -X POST http://localhost:8080/infer/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"data": [0.5, 0.5, ...], "shape": [3, 224, 224]}]}'
```

**Response:**
```json
{
  "results": [
    {
      "output": [0.12, 0.03, ...],
      "shape": [1000]
    }
  ]
}
```

### `GET /health` — Liveness Check

```json
{"status": "ok"}
```

### `GET /metrics` — Server Metrics

```json
{
  "total_requests": 1024,
  "total_batches": 256,
  "avg_batch_size": 4.0
}
```

## Benchmark Results

> Full results: [`benchmarks/results/benchmark_results.md`](benchmarks/results/benchmark_results.md)
> 
> *CPU benchmarks run on Intel/AMD x86_64 (matching 4-thread configuration). GPU benchmarks run inside the NVIDIA CUDA 11.8 Docker container.*

### Latency (Batch Size = 1)

| Engine | Mean | Median | p95 | p99 |
|--------|------|--------|-----|-----|
| **PyTorch (Python, CPU)** | 42.48 ms | 41.87 ms | 48.06 ms | 54.34 ms |
| **Helios C++ (ONNX Runtime, CPU)** | 23.82 ms | 23.70 ms | 25.72 ms | 27.46 ms |
| **Helios C++ (ONNX Runtime, GPU/CUDA)** | **3.91 ms** | **3.86 ms** | **4.45 ms** | **5.84 ms** |

### Throughput (Batch Size = 4)

| Engine | Images / sec |
|--------|-------------|
| **Helios C++ (CPU)** | 41.63 img/s |
| **Helios C++ (GPU/CUDA)** | **515.48 img/s** |

### Performance Analysis & Speedup Breakdown

- **CPU Runtime Speedup (Apples-to-Apples):** **~1.8x faster** (23.82 ms vs. 42.48 ms mean latency) than the PyTorch Python baseline on identical CPU hardware with matching thread counts (4 threads). This directly isolates the architectural performance gain from eliminating Python's GIL, runtime interpreter overhead, and serialization copies.
- **GPU Acceleration (CUDA vs. CPU C++):** **~6.1x faster** raw latency (3.91 ms vs. 23.82 ms) and **~12.4x higher throughput** (515.48 img/s vs. 41.63 img/s at batch=4).
- **Cross-Hardware Context (GPU vs. PyTorch CPU):** Helios on CUDA executes inference **~10.8x faster** than PyTorch on CPU (3.91 ms vs. 42.48 ms). *Note: While this demonstrates full-stack deployment acceleration, comparing GPU C++ directly against CPU PyTorch is a cross-hardware comparison. The fair software architectural comparison on identical hardware is the CPU-to-CPU ~1.8x speedup.*

### Running Benchmarks

```bash
# 1. Pure C++ engine benchmark (isolates raw inference latency/throughput without HTTP/JSON)
./build/Release/cpp_benchmark

# 2. PyTorch CPU baseline & HTTP server benchmark
python benchmarks/benchmark.py
```
## Testing

```bash
cd build && ctest --output-on-failure
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Inference Engine | ONNX Runtime 1.18 (C++ API) |
| GPU Acceleration | NVIDIA CUDA 11.8 + cuDNN 8 |
| HTTP Server | Crow (C++ REST framework) |
| JSON | nlohmann/json |
| Build System | CMake 3.20 + vcpkg |
| Containerization | Docker (multi-stage, NVIDIA base) |
| Testing | Google Test |
| Model | ResNet-50 (PyTorch → ONNX, dynamic batch) |

## License

MIT


