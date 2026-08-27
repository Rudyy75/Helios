#pragma once
#include "inference_engine.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>

struct InferenceRequest {
    std::vector<float> input_data;      // single sample, flattened
    std::vector<int64_t> sample_shape;  // e.g. [3, 224, 224], no batch dim
    std::promise<TensorResult> result_promise;
};

class AsyncBatcher {
public:
    AsyncBatcher(InferenceEngine& engine,
                 size_t max_batch_size = 8,
                 int batch_timeout_ms = 5);
    ~AsyncBatcher();

    AsyncBatcher(const AsyncBatcher&) = delete;
    AsyncBatcher& operator=(const AsyncBatcher&) = delete;

    std::future<TensorResult> submit(std::vector<float> input_data,
                                      std::vector<int64_t> sample_shape);

    // Metrics for the benchmark/dashboard
    size_t get_total_requests() const { return total_requests_.load(); }
    size_t get_total_batches() const { return total_batches_.load(); }
    double get_avg_batch_size() const;

private:
    void worker_loop();
    void process_batch(std::vector<std::unique_ptr<InferenceRequest>>& batch);

    InferenceEngine& engine_;
    size_t max_batch_size_;
    std::chrono::milliseconds batch_timeout_;

    std::queue<std::unique_ptr<InferenceRequest>> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::thread worker_thread_;
    std::atomic<bool> running_{true};

    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> total_batches_{0};
};
