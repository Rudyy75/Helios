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

    try {
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
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
