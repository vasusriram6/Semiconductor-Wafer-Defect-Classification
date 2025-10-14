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

def sample_with_class_coverage(df, label_col, n_samples, random_state=42):
    # Step 1: Take at least 3 sample per class to avoid training/eval errors
    base_samples = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(3, random_state=random_state)
    )

    # Step 2: Sample remaining rows randomly from the dataset
    remaining = n_samples - len(base_samples)
    if remaining > 0:
        additional = df.drop(base_samples.index).sample(
            remaining, random_state=random_state
        )
        df_sampled = pd.concat([base_samples, additional], axis=0)
    else:
        df_sampled = base_samples

    return df_sampled

train_df_sampled = sample_with_class_coverage(train_df, label_col="failureType", n_samples=120)
test_df_sampled = sample_with_class_coverage(test_df, label_col="failureType", n_samples=30)

pd.to_pickle(train_df_sampled, cfg_test["train_data_path"])
pd.to_pickle(test_df_sampled, cfg_test["test_data_path"])