import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches

df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

poly = PolynomialFeatures(5)
polydata = poly.fit_transform(X_all)
print(polydata)

# df2 = pd.DataFrame(polydata)
# df2.to_csv('out.csv', index=False)  

c_array = [0.0001, 0.001, 0.01, 1, 10, 1000]
def cToAlpha(c_array):
    alpha_array = []
    for c in c_array:
        alpha_array.append(1/2*c)

    return alpha_array

alpha_array = cToAlpha(c_array)

import numpy as np
Xtest = []
grid = np.linspace(-5,5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)


# def kFolding(all_data, labels):
#     kf = KFold(n_splits=2)
#     clf = linear_model.Lasso(alpha=0.011)
#     for train_index, test_index in kf.split(polydata):
#         # print("TRAIN:", train_index, "TEST:", test_index
#         X_train, X_test = all_data[train_index], all_data[test_index]
#         y_train, y_test = labels[train_index], labels[test_index]
#         clf.fit(X_train, y_train)
#         predictions = clf.predict(Xtest)
#         print("predictions: ", type(predictions))
#         print("y_test: ", type(y_test))
#         plt.plot(predictions)
#         # plt.plot(y_test.to_numpy())
#         plt.show()

# kFolding(polydata,X_labels)

def crossValScorePrinting(a,c):
    clf = linear_model.Lasso(alpha=a)
    scores = cross_val_score(clf, polydata, X_labels, cv=8)
    print("C:" + str(a) + " - Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


def crossValPredict():
    clf = linear_model.Lasso(alpha=0.01)
    predicted = cross_val_predict(clf, polydata, X_labels, cv=8)
    return predicted

def plot3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c='r', marker='o', label="data3.csv")
    ax.legend()
    ax.set_title("Assignment 3 Data")
    ax.set_xlabel('x1 feature', c="b")
    ax.set_ylabel('x2 feature', c="b")
    ax.set_zlabel('Y - Label', c="b")

    plt.show()



plot3D(X_all["X1"], X_all["X2"], X_labels.to_numpy())