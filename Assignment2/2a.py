# id:19-19-19
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]
X_train, X_test, y_train, y_test = train_test_split(X_all, X_labels, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
params = clf.get_params()
print(params)
b = clf.intercept_[0]
w1, w2 = clf.coef_.T

# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2
xmin, xmax = -1, 1
ymin, ymax = -1, 1
xd = np.array([xmin, xmax])
yd = m*xd + c

#Seperating 1 and -1 values into Xpositive and Xnegative
Xp = df.loc[df["X3"]==1]
Xm = df.loc[df["X3"]==-1]
Xp_labels = Xp["X3"]
Xm_labels = Xm["X3"]

predictions = clf.predict(X_test)

#splitting the data
X_predictions = X_test
X_predictions["X3"] = predictions

#splitting the data
Xp_predictions = X_predictions.loc[X_predictions["X3"]==1]
Xm_predictions = X_predictions.loc[X_predictions["X3"]==-1]
Xp_pred_labels = X_predictions["X3"]
Xm_pred_labels = X_predictions["X3"]

#plotting the data
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.tight_layout(pad=1.0)
ax1.title("Complete Data")
ax1.ylabel("x_2")
ax1.xlabel("x_1")
ax1.scatter(Xp["X1"],Xp["X2"], color='b', s = 10)
ax1.scatter(Xm["X1"],Xm["X2"], color='aqua', s = 10)
ax2.set_title("Training Data + Predicted Values")
ax2.set_ylabel("x_2")
ax2.set_xlabel("x_1")
ax2.scatter(Xp["X1"],Xp["X2"], color='b', s = 10)
ax2.scatter(Xm["X1"],Xm["X2"], color='r', s = 10)
ax2.scatter(Xp_predictions["X1"], Xp_predictions["X2"], color='c', s = 20)
ax2.scatter(Xm_predictions["X1"], Xm_predictions["X2"], color='m', s = 20)
ax3.set_title("Predicted Data + Decision Boundary (dotted)")
ax3.set_ylabel("x_2_pred")
ax3.set_xlabel("x_1_pred")
ax3.scatter(Xp_predictions["X1"], Xp_predictions["X2"], color='c')
ax3.scatter(Xm_predictions["X1"], Xm_predictions["X2"], color='m')
ax3.plot(xd, yd, 'k', lw=1, ls='--')

plt.show()