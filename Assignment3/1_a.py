# id:24--48--24
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

poly = PolynomialFeatures(5)
polydata = poly.fit_transform(X_all)
print(polydata)

fig = plt.figure()
ax = fig.gca(projection='3d')
scatter = ax.scatter(X_all["X1"], X_all["X2"], X_labels.to_numpy(), c='r', marker='o', label="data3.csv")
ax.legend()
ax.set_title("Assignment 3 Data")
ax.set_xlabel('x1 feature', c="b")
ax.set_ylabel('x2 feature', c="b")
ax.set_zlabel('Y - Label', c="b")
plt.show()