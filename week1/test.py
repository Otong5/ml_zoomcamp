import pandas as pd
import numpy as np

print(pd.__version__)
print()
df = pd.read_csv("laptops.csv")
print(df["Brand"].nunique())
print()
print(df.isna().sum())
print()
print(f"median: {df['Screen'].median()}\nmode: {df['Screen'].mode()}")
df.fillna(df["Screen"].mode())
print(f"\nprocessing...\n\nmedian: {df['Screen'].median()}")
print()
injoo = df[df["Brand"] == "Innjoo"]
X = injoo[["RAM", "Storage", "Screen"]].to_numpy()
XTX = X.T.dot(X)
inv_XTX = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100])
w = inv_XTX.dot(X.T).dot(y)
print(w.sum().round(2))
