import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import time

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


def infer_onnx(session, img_array):
    ort_inputs = {session.get_inputs()[0].name: img_array.astype(np.float32)}
    ort_outs = session.run([], ort_inputs)[0]
    output = ort_outs.flatten()  #Probability logits
    #print("ORT output is in the form of:", output)
    #probs = output

    pred_idx = int(np.argmax(output))
    confidence = float(output[pred_idx])
    # pred_idx = int(np.argmax(probs))
    # confidence = float(probs[pred_idx])

    return CLASSES[pred_idx], confidence

def load_pt_model():
    model_pt = build_model(len(CLASSES))
    state_dict = torch.load("outputs/main_output/best_model.pt", map_location="cpu")
    model_pt.load_state_dict(state_dict)
    model_pt.eval()
    
    return model_pt

def infer_pt(model_pt, img_array):
    with torch.no_grad():
        outputs = model_pt(torch.from_numpy(img_array)).numpy().squeeze() #Probability logits
        #print("PT output is in the form of:", outputs)
        pred_idx = int(np.argmax(outputs))
        confidence = float(outputs[pred_idx])

        return CLASSES[pred_idx], confidence

# Streamlit UI
st.title("Wafer Defect Classification - ONNX Inference")

uploaded_files = st.file_uploader("Upload wafer image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    session = load_onnx_session()
    print("Running ONNX inference using", session.get_providers())

    model_pt = load_pt_model()
    print("Running PyTorch inference using", next(model_pt.parameters()).device)

    results_onnx = []
    results_pt = []
    time_onnx = []
    time_pt = []

    for f in uploaded_files:
        img = Image.open(f)
        #st.image(img)  # Show the actual image you are inferring on

        train_tf, eval_tf = build_transforms(IMG_SIZE)

        input_tensor = eval_tf(img)
        img_array = input_tensor.unsqueeze(0).numpy() # create a mini-batch as expected by the model

        time1 = time.time()
        pred_class_onnx, conf_onnx = infer_onnx(session, img_array)
        time_onnx.append(time.time()-time1)

        time2 = time.time()
        pred_class_pt, conf_pt = infer_pt(model_pt, img_array)
        time_pt.append(time.time()-time2)

        results_onnx.append((f.name, pred_class_onnx, conf_onnx, img))

    st.write("Avg time taken for ONNX inference:", sum(time_onnx)/len(time_onnx))

    st.write("Avg time taken for PyTorch inference:", sum(time_pt)/len(time_pt))

    st.write("## Results:")
    for fname, pred_class, conf, img in results_onnx:
        col1, col2 = st.columns([1,4])
        with col1:
            st.image(img, width=50)
        with col2:
            st.write(f"**{fname}:** {pred_class}  (Confidence: {conf:.2f})")
