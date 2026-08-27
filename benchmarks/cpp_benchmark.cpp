#include <chrono>
#include "inference_engine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    InferenceEngine engine("models/resnet50.onnx", 4);
    std::vector<int64_t> shape = {1, 3, 224, 224};
    std::vector<float> input(3 * 224 * 224, 0.5f);

    // warmup
    for (int i = 0; i < 20; ++i) engine.run(input, shape);

    const int N = 500;
    std::vector<double> latencies;
    latencies.reserve(N);
    for (int i = 0; i < N; ++i) {
        engine.run(input, shape);
        latencies.push_back(engine.get_last_latency_ms());
    }

    std::sort(latencies.begin(), latencies.end());
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / N;

    std::cout << "Pure ONNX Runtime C++ inference (no HTTP/JSON):\n";
    std::cout << "  mean: " << mean << " ms\n";
    std::cout << "  p50:  " << latencies[N / 2] << " ms\n";
    std::cout << "  p95:  " << latencies[static_cast<size_t>(N * 0.95)] << " ms\n";
    std::cout << "  p99:  " << latencies[static_cast<size_t>(N * 0.99)] << " ms\n";

    // Batch-4 throughput comparison
    std::vector<float> batch_input(4 * 3 * 224 * 224, 0.5f);
    std::vector<int64_t> batch_shape = {4, 3, 224, 224};
    for (int i = 0; i < 10; ++i) engine.run_batch(batch_input, batch_shape);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) engine.run_batch(batch_input, batch_shape);
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\nBatch-4 throughput: " << (400.0 / (total_ms / 1000.0)) << " images/sec\n";

    return 0;
}
