# id:24--48--24
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict

df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

#Generating Xtest
def makeXTest(dimension=5):
    Xtest = []
    grid = np.linspace(-dimension,dimension)
    for i in grid:
        for j in grid:
            Xtest.append([i,j])
    Xtest = np.array(Xtest)
    return Xtest

poly = PolynomialFeatures(5)
polydata = poly.fit_transform(X_all)
print(polydata)

c_array = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.5,0.7]
for i in range(1,101):
    c_array.append(i)

print(c_array)
def cToAlpha(c_array):
    alpha_array = []
    for c in c_array:
        alpha_array.append(1/(2*c))
    return alpha_array
alpha_array = cToAlpha(c_array)

def crossValScore(a,c,cvs=10):
    clf = linear_model.Ridge(alpha=a)
    scores = cross_val_score(clf, polydata, X_labels, cv=cvs)
    return scores

ridge_mean_array = []
ridge_std_array = []
for a,c in zip(alpha_array,c_array):
    scores = crossValScore(a,c)
    print("C:" + str(c) + " - Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    ridge_mean_array.append(scores.mean())
    ridge_std_array.append(scores.std())
    # print(lasso_mean_array, lasso_std_array)

#Assignment 3 part 2
# crossFoldArray = [2, 5, 10, 25, 50, 60]
# cross_mean_array = []
# cross_std_array = []
# for folds in crossFoldArray:
#     clf = linear_model.Lasso(alpha=1/(2*10))
#     scores = cross_val_score(clf, polydata, X_labels, cv=folds)
#     print("Folds:" + str(folds) + " - Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#     cross_mean_array.append(scores.mean())
#     cross_std_array.append(scores.std())
#     # print(cross_mean_array, cross_std_array)
    
plt.errorbar(c_array, ridge_mean_array, yerr=ridge_std_array, label="Accuracy + std deviation")
# plt.errorbar(crossFoldArray, cross_mean_array, yerr=cross_std_array, label="Accuracy")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.title("Accuracy for different Cs - 10 Folds")
plt.legend()
plt.show()
