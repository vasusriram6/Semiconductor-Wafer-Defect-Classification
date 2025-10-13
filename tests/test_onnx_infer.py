import onnxruntime as ort
import numpy as np
import os
import yaml

def test_onnx_inference():
    model_path = "outputs/test_output/wafer.onnx"
    if not os.path.exists(model_path):
        return  # Skip if model not present
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    x = np.random.randn(1, 3, 128, 128).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    assert ort_outs.shape[0] == 1  # Batch size
    assert ort_outs.shape[1] >= 2  # At least 2 classes
