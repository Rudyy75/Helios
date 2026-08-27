#include "async_batcher.hpp"
#include <algorithm>

AsyncBatcher::AsyncBatcher(InferenceEngine& engine, size_t max_batch_size, int batch_timeout_ms)
    : engine_(engine),
      max_batch_size_(max_batch_size),
      batch_timeout_(batch_timeout_ms) {
    worker_thread_ = std::thread(&AsyncBatcher::worker_loop, this);
}

AsyncBatcher::~AsyncBatcher() {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

std::future<TensorResult> AsyncBatcher::submit(std::vector<float> input_data,
                                                std::vector<int64_t> sample_shape) {
    auto req = std::make_unique<InferenceRequest>();
    req->input_data = std::move(input_data);
    req->sample_shape = std::move(sample_shape);
    auto future = req->result_promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(req));
    }
    total_requests_++;
    queue_cv_.notify_one();
    return future;
}

void AsyncBatcher::worker_loop() {
    while (running_) {
        std::vector<std::unique_ptr<InferenceRequest>> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait until there's at least one request, or we're shutting down
            queue_cv_.wait(lock, [this] {
                return !request_queue_.empty() || !running_;
            });

            if (!running_ && request_queue_.empty()) break;

            auto deadline = std::chrono::steady_clock::now() + batch_timeout_;

            // Pull the first request immediately (don't wait for a full batch
            // if the queue was already non-empty when we woke up)
            while (batch.size() < max_batch_size_) {
                if (!request_queue_.empty()) {
                    batch.push_back(std::move(request_queue_.front()));
                    request_queue_.pop();
                } else {
                    // Wait for more requests up to the remaining timeout window
                    if (queue_cv_.wait_until(lock, deadline, [this] {
                            return !request_queue_.empty() || !running_;
                        })) {
                        continue; // got woken with data (or shutdown) — loop check re-evaluates
                    } else {
                        break; // timed out — send whatever we have
                    }
                }
                if (std::chrono::steady_clock::now() >= deadline) break;
            }
        }

        if (!batch.empty()) {
            process_batch(batch);
        }
    }
}

void AsyncBatcher::process_batch(std::vector<std::unique_ptr<InferenceRequest>>& batch) {
    // Stack all sample inputs into one contiguous batch tensor
    size_t batch_size = batch.size();
    size_t sample_size = batch[0]->input_data.size();

    std::vector<float> stacked_input;
    stacked_input.reserve(batch_size * sample_size);

    for (auto& req : batch) {
        if (req->input_data.size() != sample_size) {
            // Shape mismatch within a batch — fail that request individually
            req->result_promise.set_exception(
                std::make_exception_ptr(std::runtime_error("Inconsistent input size in batch")));
            continue;
        }
        stacked_input.insert(stacked_input.end(),
                              req->input_data.begin(), req->input_data.end());
    }

    std::vector<int64_t> batch_shape = { static_cast<int64_t>(batch_size) };
    for (auto d : batch[0]->sample_shape) batch_shape.push_back(d);

    try {
        TensorResult batched_result = engine_.run_batch(stacked_input, batch_shape);

        size_t per_sample_output = batched_result.data.size() / batch_size;
        for (size_t i = 0; i < batch_size; ++i) {
            TensorResult single;
            single.data.assign(
                batched_result.data.begin() + i * per_sample_output,
                batched_result.data.begin() + (i + 1) * per_sample_output);
            single.shape = std::vector<int64_t>(batched_result.shape.begin() + 1,
                                                 batched_result.shape.end());
            batch[i]->result_promise.set_value(std::move(single));
        }
        total_batches_++;
    } catch (const std::exception& e) {
        for (auto& req : batch) {
            req->result_promise.set_exception(std::current_exception());
        }
    }
}

double AsyncBatcher::get_avg_batch_size() const {
    size_t batches = total_batches_.load();
    if (batches == 0) return 0.0;
    return static_cast<double>(total_requests_.load()) / batches;
}
