# Benchmark Results

**Hardware:** CPU (Intel/AMD x86_64)
**Model:** ResNet50 (Opset 17, Dynamic Batch Axis)

## Latency Comparison (Batch Size = 1)

| Engine | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| **PyTorch (Python)** | 42.48 | 41.87 | 48.06 | 54.34 |
| **Helios C++ Core (ONNX Runtime)** | 23.82 | 23.70 | 25.72 | 27.46 |

**Speedup:** The Helios C++ Core is approximately **1.78x faster** than the PyTorch Python baseline on the same hardware, entirely eliminating Python's GIL and runtime overhead.

## Throughput (Batch Size = 4)

- **Helios C++ Core Throughput:** 41.63 images/sec

## Latency Comparison (Batch Size = 1) - GPU / CUDA
| Engine | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| **Helios C++ Core (CUDA)** | 3.91 | 3.86 | 4.45 | 5.84 |

**GPU Speedup:** The Helios C++ Core running on CUDA is approximately **9.8x faster** than the PyTorch Python CPU baseline, executing raw inference in under 4 milliseconds.

## GPU Throughput (Batch Size = 4)

- **Helios C++ Core (CUDA) Throughput:** 515.48 images/sec

