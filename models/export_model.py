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
        opset_version=17,                 # current stable opset
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
