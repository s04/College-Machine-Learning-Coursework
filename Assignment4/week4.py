# id:22-22-22-0 
# Saul O'Driscoll 17333932

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from sklearn.dummy import DummyClassifier

dummyClf = DummyClassifier(strategy = "most_frequent")

#Indicator of which dataset to use
dataset_switcher = 1

#Creating new files for the split datasets
dataset = open("data4.csv", "r")
dataset_1 = open("data4_1.csv", "w+")
dataset_2 = open("data4_2.csv", "w+")

#splitting the dataset
content = dataset.read()
content_list = content.split("# ")
dataset.close()

content_list[0] = content_list[1]
content_list[1] = content_list[2]

content_list[0] = "# " + content_list[0]
content_list[1] = "# " + content_list[1]

#writing content to the new dataframes
dataset_1.write(content_list[0])
dataset_2.write(content_list[1])
dataset_1.close()
dataset_2.close()

#reading dataset 1 from new file
df1 = pd.read_csv("data4_1.csv",header=None, comment="#", names=["X1","X2","X3"])
dataset_1_X_all = df1.drop(columns=["X3"])
dataset_1_X_labels = df1["X3"]

#reading dataset 2 from new file
df2 = pd.read_csv("data4_2.csv",header=None, comment="#", names=["X1","X2","X3"])
dataset_2_X_all = df2.drop(columns=["X3"])
dataset_2_X_labels = df2["X3"]

#assigning X and y to dataset that was selected earlier
if (dataset_switcher == 1):
    X = dataset_1_X_all
    y = dataset_1_X_labels
else:
    X = dataset_2_X_all
    y = dataset_2_X_labels

def displayDataSet():
    colors=['red' if l == -1 else 'blue' for l in y.values]
    plt.scatter(X["X1"], X["X2"], color=colors)
    plt.title("Data Week 4 - Set {0}".format(dataset_switcher))
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    red_patch = mpatches.Patch(color='red', label='-1')
    blue_patch = mpatches.Patch(color='blue', label='1')
    plt.legend(handles=[red_patch, blue_patch])

    plt.show()

# displayDataSet()

#Poly Feature Finder for seleting the best C and best Polynomial feature for logistic regression
def polyFeatureFinder():
    cArray = [0.01, 1, 2, 3, 5]
    polyArray = []

    for i in range(1,10,1):
        polyArray.append(i)

    for p in polyArray:
        polyAccuracy_mean_array = []
        polyAccuracy_std_array = []
        for c in cArray:
            poly = PolynomialFeatures(p)
            polydata = poly.fit_transform(X)

            clf = LogisticRegression(C=c)
            scores = cross_val_score(clf, polydata, y, cv=8)

            polyAccuracy_mean_array.append(scores.mean())
            polyAccuracy_std_array.append(scores.std())    

            print("C: " + str(c) + " PolyFeatures: " + str(p) +  " -> Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            
            # print(len(cArray), len(polyAccuracy_mean_array))
        plt.errorbar(cArray, polyAccuracy_mean_array, yerr=polyAccuracy_std_array, label="PolyFeat = {0}".format(p))
        
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for different C/polys - 10 Folds")
    plt.legend()
    plt.show()

    # Best score 
    """
    C: 1 PolyFeatures: 3 -> Accuracy: 0.95 (+/- 0.05)
    C: 1 PolyFeatures: 4 -> Accuracy: 0.96 (+/- 0.05)
    C: 1 PolyFeatures: 5 -> Accuracy: 0.95 (+/- 0.06)
    C: 1 PolyFeatures: 6 -> Accuracy: 0.96 (+/- 0.04)
    C: 1 PolyFeatures: 7 -> Accuracy: 0.96 (+/- 0.04)
    C: 1 PolyFeatures: 8 -> Accuracy: 0.96 (+/- 0.06)
    C: 1 PolyFeatures: 9 -> Accuracy: 0.96 (+/- 0.06)
    C: 2 PolyFeatures: 3 -> Accuracy: 0.95 (+/- 0.06)
    """

def knnKFinder():
    kArray = []
    polyArray = []

    for i in range(1,50):
        kArray.append(i)

    poly = PolynomialFeatures(6)
    polydata = poly.fit_transform(X)
    for i in range(1,6,1):
        polyArray.append(i)

    for p in polyArray:
        knnAccuracy_mean_array = []
        knnAccuracy_std_array = []
        for k in kArray:
            clf = KNeighborsClassifier(k)
    
            scores = cross_val_score(clf, polydata, y, cv=8)
            knnAccuracy_mean_array.append(scores.mean())
            knnAccuracy_std_array.append(scores.std())    

            print("K: " + str(k) + " PolyFeatures: " + str(p) + " -> Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # print(len(cArray), len(polyAccuracy_mean_array))
        plt.errorbar(kArray, knnAccuracy_mean_array, yerr=knnAccuracy_std_array, label="Poly-Features = {0}".format(p))

    # plt.errorbar(cArray, polyAccuracy_mean_array, yerr=polyAccuracy_std_array, label="Accuracy + std deviation")
    plt.xlabel("K-Neighbours")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for different K/polys - 10 Folds")
    plt.legend(loc=4)
    plt.show()
    
    #Best Scores
    """
    K: 12 -> Accuracy: 0.950 (+/- 0.03)
    K: 13 -> Accuracy: 0.952 (+/- 0.04)
    """


#Evaluate classifier with confusion matrix and accuracy score
def evaluateClf(clf, X, y):
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_Train, y_Train)

    yPred = clf.predict(X_Test)
    conf_matrix = confusion_matrix(y_Test, yPred)
    accuracy = accuracy_score(y_Test, yPred)
    return conf_matrix, accuracy

#plot ROC curve (does split data into train and test)
def ROCCurve(title,clf,X,y):
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.5)
    
    clf.fit(X_Train, y_Train)
    yPred = clf.predict(X_Test)

    plot_roc_curve(clf, X_Test, y_Test)
    plt.title(title)
    plt.plot([0,1],[0,1],color="red",linestyle="--")
    plt.show()

def bestKnnDisplay():
    k = 12 # change to 40 for dataset 2 
    clf = KNeighborsClassifier(k)
    matrix, accuracy = evaluateClf(clf, X, y)
    print("Best KNN Classifier ")
    print("Confusion Matrix:\n" + str(matrix) + " -> Accuracy: " + str(accuracy))
    matrix_dummy, dummy_accuracy = evaluateClf(dummyClf, X, y)
    print("Dummy Classifier ")
    print("Confusion Matrix:\n" + str(matrix_dummy) + " -> Accuracy: " + str(dummy_accuracy))
    ROCCurve("Top KNN Classifier", clf, X, y)

def bestLogRegDisplay():
    features = 6
    poly = PolynomialFeatures(features)
    polydata = poly.fit_transform(X)

    clf = LogisticRegression(C=1)
    matrix, accuracy = evaluateClf(clf, polydata, y)
    print("Best Logistic Regression Classifier ")
    print("Confusion Matrix:\n" + str(matrix) + " -> Accuracy: " + str(accuracy))
    matrix_dummy, dummy_accuracy = evaluateClf(dummyClf, polydata, y)
    print("Dummy Classifier ")
    print("Confusion Matrix:\n" + str(matrix_dummy) + " -> Accuracy: " + str(dummy_accuracy))
    ROCCurve("Top Logistic Regression Classifier", clf, X, y)

polyFeatureFinder()
knnKFinder()
bestKnnDisplay()
bestLogRegDisplay()