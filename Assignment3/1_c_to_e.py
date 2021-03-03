# id:24--48--24
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from matplotlib import cm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

#Importing Xtrain + Labels
df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

poly = PolynomialFeatures(5)
polydata = poly.fit_transform(X_all)

def makeXTest(dimension=5):
    Xtest = []
    grid = np.linspace(-dimension,dimension)
    for i in grid:
        for j in grid:
            Xtest.append([i,j])
    Xtest = np.array(Xtest)
    return Xtest


#Converting C to alphas for easier use
c_array = [0.001, 0.1, 1, 10, 50, 100, 1000, 10000]
c_array = [0.001, 0.01, 1, 10, 100]
def cToAlpha(c_array):
    alpha_array = []
    for c in c_array:
        alpha_array.append(1/(2*c))

    return alpha_array

alpha_array = cToAlpha(c_array)

def lassoing():
    for a,c in zip(alpha_array,c_array):
        clf = linear_model.Lasso(alpha=a)
        clf.fit(polydata, X_labels)
        Xtest = makeXTest(5)
        poly_Xtest = poly.fit_transform(Xtest)
        predictions = clf.predict(poly_Xtest)
        fig = plt.figure()
        Xtest = makeXTest(2)
        ax = fig.gca(projection='3d')
        ax.set_title("Lasso Prediction Plane with c={0} / alpha={1}".format(c, a))
        ax.set_xlabel('X1', c="b")
        ax.set_ylabel('X2', c="b")
        ax.set_zlabel('Y - Label', c="b")
        scatter = ax.scatter(X_all["X1"], X_all["X2"], X_labels, c='r', label="data3.csv")
        surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1], predictions)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        plt.legend([scatter, surf], ["Training Data", "Lasso Surface"])
        plt.show()

lassoing()

def rideRegression():
    for a,c in zip(alpha_array,c_array):
        clf = linear_model.Ridge(alpha=a)
        clf.fit(polydata, X_labels)
        Xtest = makeXTest(5)
        poly_Xtest = poly.fit_transform(Xtest)
        predictions = clf.predict(poly_Xtest)
        Xtest = makeXTest(2)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("Ridge Regression Plane with c={0} / alpha={1}".format(c, a))
        ax.set_xlabel('X1', c="b")
        ax.set_ylabel('X2', c="b")
        ax.set_zlabel('Y - Label', c="b")
        scatter = ax.scatter(X_all["X1"], X_all["X2"], X_labels, c='r', label="data3.csv")
        surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1], predictions)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        plt.legend([scatter, surf], ["Training Data", "Ridge Surface"])
        plt.show()

rideRegression()