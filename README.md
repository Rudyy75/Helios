<div align="center">

# ⚡ Helios

**High-performance C++ model inference server built on ONNX Runtime**

*Dynamic batching · CUDA GPU acceleration · REST API · Docker-ready*

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

Because Python adds overhead. Helios eliminates the Python runtime, GIL, and serialization costs entirely:

| Mode | Mean Latency | p95 Latency | Throughput |
|------|-------------|-------------|------------|
| **PyTorch CPU** (Python, batch=1) | 42.48 ms | 48.06 ms | N/A |
| **Helios C++ Core** (CPU, batch=1) | 23.82 ms | 25.72 ms | N/A |
| **Helios C++ Core** (CPU, batch=4) | N/A | N/A | 41.6 img/s |
| **Helios C++ Core** (GPU/CUDA, batch=1) | **3.91 ms** | **4.45 ms** | N/A |
| **Helios C++ Core** (GPU/CUDA, batch=4) | N/A | N/A | **515 img/s** |

> *CPU benchmarks on Intel/AMD x86_64. GPU benchmarks inside NVIDIA CUDA 11.8 Docker container.*
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
│   ├── async_batcher.hpp       # Thread-safe dynamic batcher
│   └── metrics.hpp             # Server metrics
├── src/
│   ├── inference_engine.cpp    # Core inference + CUDA EP logic
│   ├── async_batcher.cpp       # Batch queue + worker thread
│   └── server.cpp              # Crow REST API + routing
├── benchmarks/
│   ├── cpp_benchmark.cpp       # Raw C++ latency/throughput benchmark
│   └── benchmark.py            # PyTorch baseline + server comparison
├── tests/
│   ├── test_inference_engine.cpp
│   └── test_async_batcher.cpp
├── models/
│   └── export_model.py         # PyTorch → ONNX export script
├── scripts/
│   ├── setup_env.ps1           # Windows dependency setup
│   └── setup_env.sh            # Linux dependency setup
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
  -d '{"requests": [{"data": [...], "shape": [3,224,224]}, ...]}'
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
> *CPU benchmarks run on Intel/AMD x86_64. GPU benchmarks run inside the NVIDIA CUDA 11.8 Docker container.*

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

**CPU Speedup:** ~1.8x faster than PyTorch Python baseline  
**GPU Speedup:** ~10x faster than PyTorch Python baseline, ~515 images/sec throughput

### Benchmarking
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


