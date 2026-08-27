#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

struct TensorResult {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path, int num_threads = 4);

    // Single inference, input shape must match model's expected rank
    TensorResult run(const std::vector<float>& input_data,
                      const std::vector<int64_t>& input_shape);

    // True batched inference — stacks N inputs into one tensor call
    // (this is what makes batching actually fast, vs. looping run() N times)
    TensorResult run_batch(const std::vector<float>& stacked_input_data,
                            const std::vector<int64_t>& batch_input_shape);

    double get_last_latency_ms() const { return last_latency_ms_; }
    std::vector<int64_t> get_input_shape_template() const { return input_shape_template_; }
    size_t get_input_element_size() const { return input_element_count_per_sample_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // Own the strings so pointers stay valid for the session's lifetime
    std::string input_name_str_;
    std::string output_name_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    std::vector<int64_t> input_shape_template_;  // e.g. [-1, 3, 224, 224]
    size_t input_element_count_per_sample_ = 0;

    Ort::MemoryInfo memory_info_;

    // ONNX Runtime Session::Run is thread-safe for read-only inference,
    // but session_options config at construction is not reentrant.
    // Session::Run IS safe to call concurrently from multiple threads
    // as long as inputs/outputs don't alias — we still guard latency stat writes.
    std::mutex latency_mutex_;
    double last_latency_ms_ = 0.0;
};
