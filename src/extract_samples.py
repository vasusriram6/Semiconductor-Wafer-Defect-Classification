import os
import random
import numpy as np
import pandas as pd
from PIL import Image

# Configuration
PKL_PATH = "data/wafer_test.pkl"
OUTPUT_DIR = "data/extracted_samples"
NUM_SAMPLES = 100  # Change to desired number

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load test dataset
df = pd.read_pickle(PKL_PATH)

# Flatten labels (handle 1D or 2D array labels)
def flatten_label(x):
    if isinstance(x, (list, np.ndarray)):
        # 2D list with one inner list
        if len(x) == 1 and isinstance(x[0], (list, np.ndarray)):
            if len(x[0]) == 1:
                return x[0][0]
            elif len(x[0]) == 0:
                return ""  # empty inner list
            else:
                return x[0]  # unclear case, return inner list as is
        # 1D list
        elif len(x) == 1:
            return x[0]
        elif len(x) == 0:
            return ""
        else:
            return x
    else:
        # Not a list/array, return as is
        return x

df["failureType"] = df["failureType"].apply(flatten_label)
df["waferIndex"] = df["waferIndex"].astype(str)

# Filter only samples with non-empty labels
df = df[df["failureType"] != ""]

# Randomly select samples
samples = df.sample(n=NUM_SAMPLES, random_state=42)


# Save images as PNG with filename: waferIndex_defectCategory.png
for _, row in samples.iterrows():
    wafer_map = row["waferMap"]
    #wafer_index = row["waferIndex"]
    wafer_index = _
    defect = row["failureType"]

    # Convert wafer_map 2D array to 3-channel uint8 image
    img = np.array(wafer_map)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    # Normalize pixel values between 0-255
    if img.max() > img.min():
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
    pil_img = Image.fromarray(img_norm)

    filename = f"{wafer_index}_{defect}.png"
    pil_img.save(os.path.join(OUTPUT_DIR, filename))


print(f"Saved {NUM_SAMPLES} sample images to '{OUTPUT_DIR}'")