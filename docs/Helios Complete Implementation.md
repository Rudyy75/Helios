# Helios — Complete Implementation Guide

Below is a fully fleshed-out, buildable version of the plan you pasted, with the bugs fixed (there's a real lifetime bug in the original `InferenceEngine` sketch involving `AllocatedStringPtr`), the batching logic made correct and thread-safe, and exact build instructions for **Windows** (since your workspace path indicates you're on Windows) with a Linux fallback path.

Quick version note before we start: the previous ONNX Runtime release is 1.25.0, and the latest Crow release is v1.3.3, which needs at least a C++17 compiler. Use these versions instead of the 1.17.0 / older Crow versions mentioned in your original plan.

---

## 0. Project Structure (set this up first)

```
Helios/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── models/
│   ├── export_model.py
│   └── resnet50.onnx              (generated)
├── include/
│   ├── inference_engine.hpp
│   ├── async_batcher.hpp
│   └── metrics.hpp
├── src/
│   ├── inference_engine.cpp
│   ├── async_batcher.cpp
│   ├── server.cpp
│   └── metrics.cpp
├── tests/
│   ├── test_inference_engine.cpp
│   └── test_async_batcher.cpp
├── benchmarks/
│   ├── benchmark.py
│   ├── cpp_benchmark.cpp
│   └── results/
│       └── benchmark_results.md
├── third_party/                    (vcpkg or manually vendored libs)
└── scripts/
    ├── setup_env.ps1 / setup_env.sh
    └── run_server.sh
```

Create this skeleton now, before writing any code. Init git immediately:

```bash
git init
git add .
git commit -m "chore: project skeleton"
```

---

## 1. Environment Setup (Windows-first)

Since dependency management is the #1 time-sink on this kind of project, use **vcpkg** instead of manually downloading tarballs — it works cleanly for both ONNX Runtime and Crow on Windows and gives you reproducible builds.

### 1.1 Install vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

### 1.2 Install dependencies

```powershell
.\vcpkg install onnxruntime:x64-windows
.\vcpkg install crow:x64-windows
.\vcpkg install nlohmann-json:x64-windows
.\vcpkg install gtest:x64-windows      # for unit tests
```

> If `vcpkg install onnxruntime` gives you an older/CPU-only build and you specifically want GPU execution providers, fall back to the manual download from the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases) — grab the latest `onnxruntime-win-x64-<version>.zip` (not 1.17.0 — use current stable), and point CMake at it manually with `ORT_ROOT`.

### 1.3 Python environment for model export + benchmarking

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision ultralytics onnx onnxruntime requests numpy
```

---

## 2. Day 1 — Model Export + ONNX Runtime Smoke Test

### 2.1 Export the model

`models/export_model.py`:

```python
import torch
import torchvision
import onnx
import onnxruntime as ort
import numpy as np

def export_resnet50(output_path="models/resnet50.onnx"):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,                 # current stable opset, not 12
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},   # THIS is what enables real batching later
            "output": {0: "batch_size"},
        },
    )
    print(f"Exported to {output_path}")

def verify_onnx(path="models/resnet50.onnx"):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # Sanity-check inference in Python before touching C++
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    print("Output shape:", outputs[0].shape)  # should be (1, 1000)

if __name__ == "__main__":
    export_resnet50()
    verify_onnx()
```

**Critical detail your original plan missed:** set `dynamic_axes` on the batch dimension. Without this, your ONNX graph is hardcoded to batch size 1, and your entire "async batching" feature in Day 4-5 will silently break or throw shape errors when you try to run batch size > 1. Do this now, not later.

Run it:

```powershell
python models\export_model.py
```

### 2.2 Minimal C++ smoke test

`tests/smoke_test.cpp`:

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "smoke_test");
    Ort::SessionOptions session_options;

#ifdef _WIN32
    const wchar_t* model_path = L"models/resnet50.onnx";
#else
    const char* model_path = "models/resnet50.onnx";
#endif

    Ort::Session session(env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Input " << i << ": " << name.get() << " shape=[";
        for (auto d : shape) std::cout << d << " ";
        std::cout << "]\n";
    }

    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "Output " << i << ": " << name.get() << "\n";
    }

    std::cout << "Model loaded successfully.\n";
    return 0;
}
```

**Note on Windows:** ONNX Runtime's C++ API takes `const wchar_t*` for model paths on Windows and `const char*` on Linux/macOS. This trips people up constantly — handle it with a preprocessor branch as above, or use `std::filesystem::path` and `.c_str()` / `.wstring().c_str()` consistently throughout your codebase.

**Goal check for Day 1:** running the smoke test prints input shape `[batch_size(dynamic), 3, 224, 224]` and output shape info, with no exceptions thrown.

---

## 3. Day 2-3 — Core Inference Engine (fixed version)

Your original sketch has a real bug: `GetInputNameAllocated` returns an `Ort::AllocatedStringPtr`, which owns the string. Storing `input_name.get()` as a raw `const char*` in a member vector after the `AllocatedStringPtr` temporary is destroyed is **undefined behavior** — the pointer will dangle. Fix: keep the `AllocatedStringPtr` objects alive as members, or copy the strings into `std::string` and keep raw pointers into those.

`include/inference_engine.hpp`:

```cpp
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
```

`src/inference_engine.cpp`:

```cpp
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
```

### 3.1 Important threading fact you need for Day 4-5

`Ort::Session::Run` **is** safe to call from multiple threads concurrently on the same session as long as each call uses its own input/output tensors (no shared mutable state passed in). This matters because your `AsyncBatcher` design as sketched uses a single worker thread pulling from the queue — which is the simpler and safer design. Stick with one worker thread per `AsyncBatcher` instance unless you specifically want to add a thread pool of workers each with batching (more complex, optional stretch goal — see §9).

### 3.2 Unit test

`tests/test_inference_engine.cpp` (GoogleTest):

```cpp
#include <gtest/gtest.h>
#include "inference_engine.hpp"

TEST(InferenceEngineTest, LoadsModelAndRunsSingleInference) {
    InferenceEngine engine("models/resnet50.onnx", 2);
    std::vector<int64_t> shape = {1, 3, 224, 224};
    std::vector<float> input(1 * 3 * 224 * 224, 0.5f);

    auto result = engine.run(input, shape);
    EXPECT_EQ(result.data.size(), 1000);  // ImageNet classes
    EXPECT_GT(engine.get_last_latency_ms(), 0.0);
}

TEST(InferenceEngineTest, HandlesBatchSizeGreaterThanOne) {
    InferenceEngine engine("models/resnet50.onnx", 2);
    std::vector<int64_t> shape = {4, 3, 224, 224};
    std::vector<float> input(4 * 3 * 224 * 224, 0.5f);

    auto result = engine.run_batch(input, shape);
    EXPECT_EQ(result.data.size(), 4 * 1000);
    EXPECT_EQ(result.shape[0], 4);
}

TEST(InferenceEngineTest, ThrowsOnShapeMismatch) {
    InferenceEngine engine("models/resnet50.onnx", 2);
    std::vector<int64_t> shape = {1, 3, 224, 224};
    std::vector<float> bad_input(10, 0.5f);  // wrong size

    EXPECT_THROW(engine.run(bad_input, shape), std::runtime_error);
}
```

**Goal check for Days 2-3:** both single and batch-4 inference pass, latency printed is nonzero and sane (a few ms to tens of ms on CPU for ResNet50).

---

## 4. Day 4-5 — Async Request Queue + Batching (production-correct version)

The version in your plan has a subtle bug: it stores raw pointers (`InferenceRequest*`) in a `std::queue`, but never says who owns/frees them, and the `process_batch` function signature is declared but not shown filled in — the actual stacking logic (concatenating N separate input vectors into one contiguous batch tensor) is the hard part and was skipped. Here's the complete, correct version.

`include/async_batcher.hpp`:

```cpp
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
```

`src/async_batcher.cpp`:

```cpp
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
```

### 4.1 Why this version is correct where the sketch wasn't

- **Ownership is explicit**: `std::unique_ptr<InferenceRequest>` makes it clear the queue owns requests until they're moved into the batch vector — no leaks, no dangling pointers.
- **Real batch stacking**: the sketch never showed how to combine N separate `[3,224,224]` inputs into one `[N,3,224,224]` tensor call. This is the actual "batching" optimization — without it you're just running N sequential single inferences and calling it a batcher.
- **Output splitting**: after a batched ONNX Runtime call returns a `[N, 1000]` tensor, each waiting `promise` needs its own `[1000]` slice — handled above.
- **Timeout logic uses `wait_until` with a real deadline** instead of a fixed sleep, so latency stays bounded even under bursty load.

### 4.2 Unit test for the batcher

```cpp
#include <gtest/gtest.h>
#include "async_batcher.hpp"
#include "inference_engine.hpp"

TEST(AsyncBatcherTest, HandlesConcurrentRequestsCorrectly) {
    InferenceEngine engine("models/resnet50.onnx", 4);
    AsyncBatcher batcher(engine, 8, 10);

    std::vector<int64_t> sample_shape = {3, 224, 224};
    std::vector<std::future<TensorResult>> futures;

    for (int i = 0; i < 20; ++i) {
        std::vector<float> input(3 * 224 * 224, 0.1f * i);
        futures.push_back(batcher.submit(input, sample_shape));
    }

    for (auto& f : futures) {
        auto result = f.get();
        EXPECT_EQ(result.data.size(), 1000);
    }

    EXPECT_EQ(batcher.get_total_requests(), 20);
    EXPECT_GT(batcher.get_avg_batch_size(), 1.0);  // proves batching actually happened
}
```

That last assertion (`avg_batch_size > 1.0`) is your proof-of-correctness for the whole feature — if this fails, requests are being processed one at a time regardless of what your batching code claims to do.

**Goal check for Days 4-5:** 20 concurrent submits complete correctly, and `avg_batch_size` is meaningfully above 1.0 (ideally 3-8 depending on your timeout window and how fast you submit them).

---

## 5. Day 6-7 — REST API Layer (Crow, current API)

Crow v1.3.x uses a slightly different CMake target name than older versions. Full working `CMakeLists.txt` first, then the server.

### 5.1 CMakeLists.txt

```cmake
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
```

Build:

```powershell
mkdir build; cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

### 5.2 Server implementation (with real input validation + error handling)

`src/server.cpp`:

```cpp
#include "crow.h"
#include "async_batcher.hpp"
#include "inference_engine.hpp"
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    std::string model_path = argc > 1 ? argv[1] : "models/resnet50.onnx";
    int num_threads = argc > 2 ? std::stoi(argv[2]) : 4;
    size_t max_batch_size = argc > 3 ? std::stoul(argv[3]) : 8;
    int batch_timeout_ms = argc > 4 ? std::stoi(argv[4]) : 5;

    InferenceEngine engine(model_path, num_threads);
    AsyncBatcher batcher(engine, max_batch_size, batch_timeout_ms);

    crow::SimpleApp app;

    CROW_ROUTE(app, "/health")([]() {
        json resp = {{"status", "ok"}};
        return crow::response(200, resp.dump());
    });

    CROW_ROUTE(app, "/metrics")([&batcher]() {
        json resp = {
            {"total_requests", batcher.get_total_requests()},
            {"total_batches", batcher.get_total_batches()},
            {"avg_batch_size", batcher.get_avg_batch_size()}
        };
        return crow::response(200, resp.dump());
    });

    CROW_ROUTE(app, "/infer").methods("POST"_method)(
        [&batcher](const crow::request& req) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception&) {
                return crow::response(400, R"({"error":"invalid JSON"})");
            }

            if (!body.contains("data") || !body.contains("shape")) {
                return crow::response(400, R"({"error":"missing 'data' or 'shape' field"})");
            }

            std::vector<float> input_data;
            std::vector<int64_t> input_shape;
            try {
                input_data = body["data"].get<std::vector<float>>();
                input_shape = body["shape"].get<std::vector<int64_t>>();
            } catch (const std::exception&) {
                return crow::response(400, R"({"error":"malformed 'data' or 'shape'"})");
            }

            size_t expected = 1;
            for (auto d : input_shape) expected *= static_cast<size_t>(d);
            if (expected != input_data.size()) {
                return crow::response(422, R"({"error":"data size does not match shape"})");
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto future = batcher.submit(std::move(input_data), std::move(input_shape));

            try {
                auto result = future.get();  // blocks this handler's thread, not others (Crow is multithreaded)
                auto end = std::chrono::high_resolution_clock::now();
                double latency = std::chrono::duration<double, std::milli>(end - start).count();

                json response;
                response["output"] = result.data;
                response["shape"] = result.shape;
                response["latency_ms"] = latency;
                return crow::response(200, response.dump());
            } catch (const std::exception& e) {
                json err = {{"error", e.what()}};
                return crow::response(500, err.dump());
            }
        });

    CROW_ROUTE(app, "/infer/batch").methods("POST"_method)(
        [&batcher](const crow::request& req) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception&) {
                return crow::response(400, R"({"error":"invalid JSON"})");
            }

            if (!body.contains("requests") || !body["requests"].is_array()) {
                return crow::response(400, R"({"error":"missing 'requests' array"})");
            }

            std::vector<std::future<TensorResult>> futures;
            for (auto& item : body["requests"]) {
                futures.push_back(batcher.submit(
                    item["data"].get<std::vector<float>>(),
                    item["shape"].get<std::vector<int64_t>>()));
            }

            json response;
            response["results"] = json::array();
            for (auto& f : futures) {
                try {
                    auto result = f.get();
                    response["results"].push_back({{"output", result.data}, {"shape", result.shape}});
                } catch (const std::exception& e) {
                    response["results"].push_back({{"error", e.what()}});
                }
            }
            return crow::response(200, response.dump());
        });

    std::cout << "Helios server listening on :8080\n";
    app.port(8080).multithreaded().run();
    return 0;
}
```

Test it manually:

```powershell
# health check
curl http://localhost:8080/health

# generate a random 3x224x224 float array in Python and POST it — see benchmark.py below
```

**Goal check for Days 6-7:** `/health`, `/infer`, `/infer/batch`, and `/metrics` all respond correctly via curl/Postman; sending 20 concurrent requests via a quick Python script shows `avg_batch_size > 1` on `/metrics`.

---

## 6. Day 8-9 — Benchmarking (do this rigorously — it's your strongest resume evidence)

### 6.1 Python comparison script (fixed + expanded)

`benchmarks/benchmark.py`:

```python
import torch
import torchvision
import time
import requests
import numpy as np
import json
import statistics
import concurrent.futures

MODEL_PATH_INFO = "ResNet50, opset 17, dynamic batch axis"
SERVER_URL = "http://localhost:8080"

def benchmark_pytorch(n=200, batch_size=1):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    torch.set_num_threads(4)  # match ONNX Runtime's thread count for fair comparison

    dummy = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(n):
            start = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - start) * 1000)

    return latencies

def benchmark_server_single(n=200):
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    payload = {"data": dummy.flatten().tolist(), "shape": [1, 3, 224, 224]}

    for _ in range(20):
        requests.post(f"{SERVER_URL}/infer", json=payload)

    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        r = requests.post(f"{SERVER_URL}/infer", json=payload)
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies

def benchmark_server_throughput(num_requests=500, num_workers=16):
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    payload = {"data": dummy.flatten().tolist(), "shape": [1, 3, 224, 224]}

    def send(_):
        return requests.post(f"{SERVER_URL}/infer", json=payload)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send, i) for i in range(num_requests)]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - start

    resp = requests.get(f"{SERVER_URL}/metrics").json()
    return {
        "throughput_rps": num_requests / elapsed,
        "elapsed_s": elapsed,
        "avg_batch_size": resp["avg_batch_size"],
    }

def summarize(name, latencies):
    print(f"\n--- {name} ---")
    print(f"  mean:   {statistics.mean(latencies):.2f} ms")
    print(f"  median: {statistics.median(latencies):.2f} ms")
    print(f"  p95:    {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
    print(f"  p99:    {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
    print(f"  min:    {min(latencies):.2f} ms")
    print(f"  max:    {max(latencies):.2f} ms")

if __name__ == "__main__":
    print(f"Model: {MODEL_PATH_INFO}")

    pt_lat = benchmark_pytorch(n=200)
    summarize("PyTorch CPU baseline", pt_lat)

    onnx_lat = benchmark_server_single(n=200)
    summarize("Helios (single request, ONNX Runtime C++)", onnx_lat)

    throughput = benchmark_server_throughput(num_requests=500, num_workers=16)
    print("\n--- Concurrent throughput (16 threads, 500 requests) ---")
    print(json.dumps(throughput, indent=2))

    speedup = statistics.mean(pt_lat) / statistics.mean(onnx_lat)
    print(f"\nSpeedup vs PyTorch: {speedup:.2f}x")
```

**Important fairness notes to write into your README** (this is what separates a serious systems project from a toy one):

1. **Thread count must match.** `torch.set_num_threads(4)` should equal `SetIntraOpNumThreads(4)` in the server, or the comparison is meaningless.
2. **The Python HTTP roundtrip includes network + JSON serialization overhead** that the raw PyTorch call doesn't. Report both "wall clock via HTTP" and, separately, a pure C++ benchmark harness (below) that measures ONNX Runtime latency with no HTTP/JSON layer, so readers can see how much of the latency is inference vs. serving overhead.
3. Report percentiles (p50/p95/p99), not just mean — that's what real infra engineers look at, and it's an easy way to show you understand production benchmarking, not just toy scripts.

### 6.2 Pure C++ benchmark harness (isolates inference-only latency)

`benchmarks/cpp_benchmark.cpp`:

```cpp
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
```

Add this as a target in CMakeLists (mirrors `unit_tests` target).

### 6.3 Memory profiling (mentioned in your plan but never specified how)

On Windows, use **Visual Studio's built-in Diagnostic Tools** (Debug → Performance Profiler → Memory Usage) while the server runs under load. On Linux, `valgrind --tool=massif` or simply `/usr/bin/time -v` for peak RSS. Record:
- Idle memory (model loaded, no requests)
- Peak memory under 16-thread concurrent load
- Whether `EnableCpuMemArena()` measurably reduces allocator churn (test with it on vs. off — this gives you a real, defensible number instead of a hand-wave claim)

Put all of this into `benchmarks/results/benchmark_results.md` with a results table, not just console output — this is what goes into your README.

**Goal check for Days 8-9:** you have real p50/p95/p99 latency numbers for PyTorch, HTTP-server ONNX, and pure-C++ ONNX, plus a throughput number and a memory number, all in a markdown table with the exact hardware you tested on documented (CPU model, thread count, OS).

---

## 7. Day 10 — README + GitHub polish

Structure your README around what a hiring manager or interviewer actually reads in 90 seconds:

```markdown
# Helios

A production-style C++ inference server built on ONNX Runtime, with 
async request batching, a Crow REST API, and benchmarked performance
against a PyTorch baseline.

## Architecture

Client (HTTP/JSON)
   │
   ▼
Crow REST API (/infer, /infer/batch, /health, /metrics)
   │
   ▼
AsyncBatcher (thread-safe queue, timeout-based batch formation)
   │
   ▼
InferenceEngine (ONNX Runtime C++ session, memory arena, graph opt)
   │
   ▼
ResNet50 (ONNX, dynamic batch axis)

## Benchmarks (Intel/AMD CPU model here, N threads, OS)

| Mode                                | Mean (ms) | p95 (ms) | Throughput |
|--------------------------------------|-----------|----------|------------|
| PyTorch CPU baseline (batch=1)       |           |          |            |
| ONNX Runtime, pure C++ (batch=1)     |           |          |            |
| ONNX Runtime, pure C++ (batch=4)     |           |          | img/sec    |
| Helios server, single req            |           |          |            |
| Helios server, 16 concurrent         |           |          | req/sec    |

## Key Engineering Decisions
- Fixed dynamic batch axis at export time to enable true batched inference
- Thread-safe async batcher using promise/future, condition_variable with
  deadline-based `wait_until` instead of polling
- Memory arena + graph optimization (ORT_ENABLE_ALL) in ONNX Runtime session
- String lifetime management for ONNX Runtime I/O name pointers (a common
  source of UB in naive ORT C++ integrations)

## Build & Run
[vcpkg + cmake instructions]

## API Reference
POST /infer        — single sample inference
POST /infer/batch   — client-side batch of independent samples
GET  /health        — liveness check
GET  /metrics       — total requests, total batches formed, avg batch size

## Testing
`ctest` runs unit tests covering shape validation, batch correctness,
and concurrent request handling.
```

Also add a `.gitignore` that excludes `build/`, `venv/`, `*.onnx` (or keep a small model in git if size allows and note the export command instead), and `vcpkg_installed/`.

Push:

```bash
git add .
git commit -m "feat: complete Helios inference server with async batching + benchmarks"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

---

## 8. Full Checklist Before You Call It Done

- [ ] ONNX export uses dynamic batch axis, verified in Python before touching C++
- [ ] C++ smoke test loads model, prints correct I/O shapes
- [ ] `InferenceEngine` unit tests pass (single, batch-4, shape-mismatch throws)
- [ ] `AsyncBatcher` unit test proves `avg_batch_size > 1.0` under concurrent load
- [ ] Server responds correctly to `/health`, `/infer`, `/infer/batch`, `/metrics`, with proper 400/422/500 error codes on bad input
- [ ] Benchmark results include p50/p95/p99 for at least 3 configurations, on documented hardware
- [ ] Memory usage measured with and without arena allocator enabled
- [ ] README has architecture diagram, real numbers (no placeholders), and clearly explains engineering tradeoffs
- [ ] Code is on GitHub with a clean commit history (not one giant commit)

## 9. Optional Stretch Goals (if you finish early)

- Add a `/infer` GPU path via `CUDAExecutionProvider` and benchmark CPU vs GPU
- Multiple worker threads in `AsyncBatcher`, each with its own `ONNX Runtime` session sharing weights via `Ort::Env` — tests your understanding of when sharing sessions across threads is/isn't safe
- Prometheus metrics endpoint instead of the custom JSON `/metrics` — a very "production" touch for a resume
- Dockerfile so the whole thing runs with `docker run` — significantly increases perceived production-readiness for an interviewer skimming your repo

This version fixes the three real bugs in your original sketch (dangling `AllocatedStringPtr`, missing dynamic batch axis at export, and unspecified batch-stacking logic), and gives you copy-pasteable, buildable code for every remaining piece.