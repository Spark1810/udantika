# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
train = pd.read_parquet(path + "train_meta.parquet")

# %%
train.head()
# %%
train_label = pd.read_csv(path + "train_labels.csv")
train_label.head()
# %%
train[["customer_ID", "preds"]].merge(train_label, how="left", on="customer_ID").head()
# %%
train["preds"] = np.round(train["preds"], 1)
train.tail()
# %%
from sklearn.preprocessing import LabelEncoder

# LabelEncoder 객체 생성한 후, fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print("인코딩 변화값: ", labels)
# %%
train[train["preds"] > 0.0]
# %%
