import pandas as pd
df = pd.read_pickle("data/wafer_train_unittest.pkl")
print(df.head())

with open("data/wafer_test_unittest.pkl", "rb") as f:
    print(f.read(100))