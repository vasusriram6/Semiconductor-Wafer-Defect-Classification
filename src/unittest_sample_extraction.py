import pandas as pd
import numpy as np
import yaml
import os

    
cfg_main_file = "configs/config.yaml"
with open(cfg_main_file, "r") as f:
    cfg_main = yaml.safe_load(f)

cfg_test_file = "configs/config_test.yaml"
with open(cfg_test_file, "r") as f:
    cfg_test = yaml.safe_load(f)


df = pd.read_pickle(cfg_main["data_path"])


#Correcting typos if any
if 'trianTestLabel' in df and 'trainTestLabel' not in df:
    df['trainTestLabel']=df['trianTestLabel']
    df.drop('trianTestLabel', axis=1, inplace=True)
elif 'trianTestLabel' in df and 'trainTestLabel' in df:
    df.drop('trianTestLabel', axis=1, inplace=True)

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
    
# Extract relevant data points
# Flatten trainTestLabel and failureType arrays: e.g. [[Training]] -> "Training"
df["trainTestLabel"] = df["trainTestLabel"].apply(flatten_label)
df["failureType"] = df["failureType"].apply(flatten_label)

# Filter only Training data for training/validation split. 
# Note:I swapped Training and Test since Test has more samples than Training
train_df = df[df["trainTestLabel"] == "Test"].reset_index(drop=True)
test_df = df[df["trainTestLabel"] == "Training"].reset_index(drop=True)

# Get unique classes and class to index mapping
classes = sorted(train_df["failureType"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}

# Map labels to integers
train_df["label_idx"] = train_df["failureType"].map(class_to_idx)
test_df["label_idx"] = test_df["failureType"].map(class_to_idx)

pd.to_pickle(train_df.iloc[0:4], cfg_test["train_data_path"])
pd.to_pickle(test_df.iloc[0:2], cfg_test["test_data_path"])