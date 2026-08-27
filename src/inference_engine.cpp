#include "inference_engine.hpp"
#include <chrono>
#include <numeric>
#include <stdexcept>

InferenceEngine::InferenceEngine(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.EnableMemPattern();
    session_options_.EnableCpuMemArena();
    // Sequential execution is usually faster for single-graph CNNs like ResNet;
    // parallel execution mode helps more for models with independent branches.
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef USE_CUDA
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const std::exception& e) {
        // Fall back to CPU if CUDA provider fails
    }
#endif

#ifdef _WIN32
    std::wstring wide_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wide_path.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session_->GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session_->GetOutputNameAllocated(0, allocator);

    input_name_str_ = input_name_alloc.get();
    output_name_str_ = output_name_alloc.get();

    input_names_ = { input_name_str_.c_str() };
    output_names_ = { output_name_str_.c_str() };

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_template_ = tensor_info.GetShape();

    // Compute per-sample element count, treating dynamic (-1) dims as 1
    // for the purpose of this calculation — actual batch dim is handled at call time
    size_t count = 1;
    for (size_t i = 1; i < input_shape_template_.size(); ++i) {
        count *= static_cast<size_t>(input_shape_template_[i]);
    }
    input_element_count_per_sample_ = count;
}

TensorResult InferenceEngine::run(const std::vector<float>& input_data,
                                   const std::vector<int64_t>& input_shape) {
    return run_batch(input_data, input_shape);
}

TensorResult InferenceEngine::run_batch(const std::vector<float>& stacked_input_data,
                                         const std::vector<int64_t>& batch_input_shape) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t expected_size = 1;
    for (auto d : batch_input_shape) expected_size *= static_cast<size_t>(d);
    if (expected_size != stacked_input_data.size()) {
        throw std::runtime_error(
            "Input data size (" + std::to_string(stacked_input_data.size()) +
            ") does not match shape (" + std::to_string(expected_size) + ")");
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(stacked_input_data.data()),
        stacked_input_data.size(),
        batch_input_shape.data(),
        batch_input_shape.size());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1);

    auto end = std::chrono::high_resolution_clock::now();
    double latency = std::chrono::duration<double, std::milli>(end - start).count();
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        last_latency_ms_ = latency;
    }

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    TensorResult result;
    result.data.assign(output_data, output_data + output_size);
    result.shape = output_shape;
    return result;
}

