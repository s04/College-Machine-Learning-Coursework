# id:24--48--24
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from matplotlib import cm
from sklearn.model_selection import cross_val_score, cross_val_predict
# np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
#Importing Xtrain + Labels
df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

poly = PolynomialFeatures(5)
polydata = poly.fit_transform(X_all)

#Converting C to alphas for easier use
c_array = [0.001, 0.01, 1, 2, 5, 10, 100, 1000]
c_array = [0.1, 1, 10, 50, 1000]
def cToAlpha(c_array):
    alpha_array = []
    for c in c_array:
        alpha_array.append(1/(2*c))

    return alpha_array
alpha_array = cToAlpha(c_array)

def lassoing(a, c):
    clf1 = linear_model.Lasso(alpha=a)
    cross_val_score(clf1, polydata, X_labels, cv=8)
    clf1.fit(polydata, X_labels)
    print(clf1.sparse_coef_)
    coefs = clf1.coef_
    intercept = clf1.intercept_
    # print("Coefficients = ", coefs)
    print("Intercept = ", intercept)    
    # Xtest = makeXTest(5)
    # poly_Xtest = poly.fit_transform(Xtest)
    # predictions = clf.predict(poly_Xtest)
    # fig = plt.figure()
    # Xtest = makeXTest(2)
    # ax = fig.gca(projection='3d')
    # ax.set_title("Lasso Prediction Plane with c={0} / alpha={1}".format(c, a))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.scatter(X_all["X1"], X_all["X2"], X_labels, c='r')
    # surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1], predictions)
    # plt.show()


def crossValScorePrinting(a,c,cvs=8):
    clf = linear_model.Lasso(alpha=a)
    scores = cross_val_score(clf, polydata, X_labels, cv=cvs)
    # lasso_mean_array = []
    # lasso_std_array = []
    print("C: " + str(c) + " Alpha: " + str(a) +  " -> Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # lasso_mean_array.append(scores.mean())
    # lasso_std_array.append(scores.std())
    # print(lasso_mean_array, lasso_std_array)
    # return scores

for a,c in zip(alpha_array,c_array):
    print("--------------------------------------------")
    crossValScorePrinting(a, c)
    lassoing(a, c)
