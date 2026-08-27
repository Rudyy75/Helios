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
