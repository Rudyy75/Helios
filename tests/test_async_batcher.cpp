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
