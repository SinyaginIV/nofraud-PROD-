import pandas as pd

train = pd.read_csv("train_fraud_dataset.csv")
test = pd.read_csv("test_no_fraud_dataset.csv")
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())
print("train dtypes:", train.dtypes)
print("test dtypes:", test.dtypes)