import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data2.csv",comment='#')
# print(df.head())
X = np.array(df.iloc[:,0])
X = X.reshape(-1,1)
y = np.array(df.iloc[:,1])
y = y.reshape(-1,1)

norm = []

#normalising the data
ymin, ymax = min(y), max(y)
for i, val in enumerate(y):
    norm.append((val-ymin) / (ymax-ymin))

reg = LinearRegression().fit(X, norm)
m = reg.coef_[0][0]
b = reg.intercept_[0]
print(m,b)
plt.scatter(X, norm, color='green')
plt.plot(X, m*X+b, color = "black")
plt.ylabel('normalized data')
plt.xlabel('number of points')
plt.gca().set_title("Sklearn LINREG")

plt.show()