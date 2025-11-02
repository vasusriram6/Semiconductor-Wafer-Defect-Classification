import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

from models import build_model # ensure import path
from transforms import build_transforms

# Configuration
ONNX_MODEL_PATH = "outputs/main_output/wafer.onnx"
IMG_SIZE = 128

# Load class names following the exact order
CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"
]


# ONNX inference helper
#@st.cache(allow_output_mutation=True)
@st.cache_resource
def load_onnx_session():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

def softmax(logits):
    # Step 1: Exponentiate the logits to scale them
    exp_values = np.exp(logits - np.max(logits))  # Subtracting max for numerical stability
    # Step 2: Normalize the exponentials by dividing with their sum
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def infer_onnx(session, img_array):
    ort_inputs = {session.get_inputs()[0].name: img_array.astype(np.float32)}
    ort_outs = session.run([], ort_inputs)[0]
    output = ort_outs.flatten()   #Probability logits
    probs = softmax(output)
    print(probs)

    pred_idx = int(np.argmax(output))
    confidence = probs[pred_idx]
    # pred_idx = int(np.argmax(probs))
    # confidence = float(probs[pred_idx])

    # Uncomment the below commented code for verifying consistency between pytorch and ONNX models

    # top3_waferid = np.argsort(-probs)[:3]
    # for waferid in top3_waferid:
    #     print(waferid, CLASSES[waferid], output[waferid])

    # print("ONNX model input name:", session.get_inputs()[0].name)
    # print("ONNX model input shape:", session.get_inputs()[0].shape)
    
    # model_pt = build_model(len(CLASSES))

    # sd = torch.load("outputs/main_output/best_model.pt", map_location="cpu")

    # model_pt.load_state_dict(sd); model_pt.eval()

    # with torch.no_grad():
    #     logits_pt = model_pt(torch.from_numpy(img_array)).numpy()

    # print("PT logits:", logits_pt); print("ONNX logits:", ort_outs)

    return CLASSES[pred_idx], confidence

# Streamlit UI
st.title("Wafer Defect Classification - ONNX Inference")

uploaded_files = st.file_uploader("Upload wafer image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    session = load_onnx_session()
    print("Running ONNX inference using", session.get_providers())
    results = []
    for f in uploaded_files:
        img = Image.open(f)
        #st.image(img)  # Show the actual image you are inferring on

        train_tf, eval_tf = build_transforms(IMG_SIZE)

        input_tensor = eval_tf(img)
        img_array = input_tensor.unsqueeze(0).numpy() # create a mini-batch as expected by the model

        pred_class, conf = infer_onnx(session, img_array)
        results.append((f.name, pred_class, conf, img))

    st.write("## Results:")
    for fname, pred_class, conf, img in results:
        col1, col2 = st.columns([1,4])
        with col1:
            st.image(img, width=50)
        with col2:
            st.write(f"**{fname}:** {pred_class}  (Confidence: {conf:.6f})")
