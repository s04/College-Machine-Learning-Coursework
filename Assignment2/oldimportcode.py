oldimportcode.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
# col_one_list = df[0].tolist()
print(df.head())
# df['X3'] = df['X3'].apply(lambda x: True if x == 1 else False)

# mask = pd.array(df["X3"], dtype="boolean")
X_plus = df.loc[df['X3']==1]
# X_minus = df[not df['X3']]
print(X_plus.head())


# clf = LogisticRegression(random_state=0).fit(X_plus, X_plus)

# plt.scatter(X1,X2, color='black')
# plt.scatter(X_minus[0], X_minus[1], color='green')
# plt.show()