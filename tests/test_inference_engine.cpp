#include <gtest/gtest.h>
#include "inference_engine.hpp"

TEST(InferenceEngineTest, LoadsModelAndRunsSingleInference) {
#ifdef _WIN32
    std::string model_path = "models/resnet50.onnx";
#else
    std::string model_path = "models/resnet50.onnx";
#endif
    InferenceEngine engine(model_path, 2);
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
