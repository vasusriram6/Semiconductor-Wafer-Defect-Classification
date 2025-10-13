import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

def load_and_process_pandas_data(data_path, train_data_path, test_data_path, seed=42):
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):

        print("Train test split dataset pickle files already available, using them.")
        train_df = pd.read_pickle(train_data_path)
        test_df = pd.read_pickle(test_data_path)
        classes = sorted(train_df["failureType"].unique())
        print("Read pickles successfully.")

    else:
        
        print("Train test splits not found, reading the entire pickle dataset.")
        df = pd.read_pickle(data_path)
        print("Read pickle successfully")

        #Correcting typos if any
        if 'trianTestLabel' in df and 'trainTestLabel' not in df:
            df['trainTestLabel']=df['trianTestLabel']
            df.drop('trianTestLabel', axis=1, inplace=True)
        elif 'trianTestLabel' in df and 'trainTestLabel' in df:
            df.drop('trianTestLabel', axis=1, inplace=True)
        
        # Flatten trainTestLabel and failureType arrays
        df["trainTestLabel"] = df["trainTestLabel"].apply(flatten_label)
        df["failureType"] = df["failureType"].apply(flatten_label)

        # Filter only Training data for training/validation split. Note:I swapped Training and Test since Test has more samples than Training
        train_df = df[df["trainTestLabel"] == "Test"].reset_index(drop=True)
        test_df = df[df["trainTestLabel"] == "Training"].reset_index(drop=True)

        classes = sorted(train_df["failureType"].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}

        train_df["label_idx"] = train_df["failureType"].map(class_to_idx)
        test_df["label_idx"] = test_df["failureType"].map(class_to_idx)

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_df["waferMap"].values, train_df["label_idx"].values,
        test_size=0.2, stratify=train_df["label_idx"].values, random_state=seed)

    test_imgs = test_df["waferMap"].values
    test_lbls = test_df["label_idx"].values

    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls), classes


class WaferMapDataset(Dataset):
    def __init__(self, wafer_maps, labels, transform=None):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.wafer_maps[idx]
        img = np.array(img)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.max() > img.min():
            img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img, dtype=np.uint8)
        img = Image.fromarray(img_norm)

        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]