# id:2--6--6
# Saul O'Driscoll 17333932

import random
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score

# Reading in data from the file
df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2"])
dataset_1_X_all = df["X1"]
dataset_1_X_labels = df["X2"]

#Opening log file
file1 = open("coef_data.txt","w+") 

# change this line to switch between dummy data and the real data
dummy = False

if dummy:
    data_points = 3
    test_plane = 100
    Xtrain = [-1, 0, 1]
    ytrain = [0, 1, 0]
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xtrain = Xtrain.reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)
else: 
    data_points = 874
    test_plane = 999
    Xtrain = np.array(dataset_1_X_all)
    ytrain = np.array(dataset_1_X_labels)
    Xtrain = Xtrain.reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)

# creating test space
Xtest = np.linspace(-3,3,num=test_plane).reshape(-1, 1)

# Defining different gaussian kernels
def gaussian_kernel0(distances):
    weights = np.exp(0*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel1(distances):
    weights = np.exp(-1*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel5(distances):
    weights = np.exp(-5*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel10(distances):
    weights = np.exp(-10*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel25(distances):
    weights = np.exp(-25*(distances**2))
    return weights/np.sum(weights)

kernel_array = [gaussian_kernel0, gaussian_kernel1, gaussian_kernel5, gaussian_kernel10, gaussian_kernel25]

# KNN Regressor 
knnAccuracy_mean_array = []
knnAccuracy_std_array = []
for g in kernel_array:
    model = KNeighborsRegressor(n_neighbors=data_points,weights=g)
z
    
    model.fit(Xtrain, ytrain)
    if(not dummy):
        scores = cross_val_score(model, Xtrain, ytrain, cv=8)
        knnAccuracy_mean_array.append(scores.mean())
        knnAccuracy_std_array.append(scores.std())   
        file1.write("Kernel: " + str(g.__name__) + " ==> Accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std() * 2))

    ypred = model.predict(Xtest)    
    plt.rc('font', size=12); plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(Xtrain, ytrain, color='red', marker='+')
    plt.plot(Xtest, ypred, color='green')
    plt.xlabel("input x"); plt.ylabel("output y")
    plt.title("KNN Regressor, kernel = {0}".format(g.__name__))
    plt.legend(["predict","train"])
    
    plt.show()
    plt.clf()

if(not dummy):
    print("KNN MEAN ARRAY")
    print(knnAccuracy_mean_array)

    print("KNN STD ARRAY")
    print(knnAccuracy_std_array)


#Parameters for kernel Ridge Regression
c_array = [0.001, 0.1, 1, 1000]
gamma_array = [0, 1, 5, 10, 25]
colour_array = ['red', 'green', 'blue', 'black', 'pink']

# Writing to log file
file1.write("Kernel Ridge Regression, C iteration \n")

# kernel Ridge Regression with Cs as the outer loop

for c in c_array:
    for g, colour in zip(gamma_array, colour_array):
        # Train and predict
        clf = KernelRidge(kernel = 'rbf', alpha=1/(2*c), gamma=g)
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)

        #Writing to log file
        # file1.write("C = {0}, Gamma = {1} \n".format(c, g))
        # file1.write("coefficients = {0} \n\n".format(clf.dual_coef_))

        #Graph printing
        plt.rc('font', size=12); plt.rcParams['figure.constrained_layout.use'] = True
        plt.scatter(Xtrain, ytrain, color='red', marker='+')
        plt.plot(Xtest, ypred, color=colour)
        plt.xlabel("input x"); plt.ylabel("output y"); plt.legend(["predict","train"])
        
        #Legend / Label loop
        patches = []
        for g, colour in zip(gamma_array, colour_array):
            patches.append(mpatches.Patch(color=colour, label='gamma = {0}'.format(g)))  
        plt.title("Kernel Ridge Regression, c = {:.3f}".format(c))
        plt.legend(handles=patches)
    plt.show()
    plt.clf()

ridgeAccuracy_mean_array = []
ridgeAccuracy_std_array = []
# kernel Ridge Regression with Gammas as the outer loop
for g in gamma_array:
    for c, colour in zip(c_array, colour_array):
        # Train and predict
        clf = KernelRidge(kernel = 'rbf', alpha=1/(2*c), gamma=g)
        clf.fit(Xtrain, ytrain)

        if(not dummy):
            scores = cross_val_score(clf, Xtrain, ytrain, cv=8)
            ridgeAccuracy_mean_array.append(scores.mean())
            ridgeAccuracy_std_array.append(scores.std())   
            file1.write("Gamma: " + str(g) + " C: " + str(c) + "-> Accuracy: %0.3f (+/- %0.3f) \n" % (scores.mean(), scores.std() * 2))

        ypred = clf.predict(Xtest)
        #Graph printing
        plt.rc('font', size=12); plt.rcParams['figure.constrained_layout.use'] = True
        plt.scatter(Xtrain, ytrain, color='red', marker='+')
        plt.plot(Xtest, ypred, color=colour)
        plt.xlabel("input x"); plt.ylabel("output y"); plt.legend(["predict","train"])
        
        #Legend / Label loop
        patches = []
        for c, colour in zip(c_array, colour_array):
            patches.append(mpatches.Patch(color=colour, label='C = {0}'.format(c)))  
        plt.title("Kernel Ridge Regression, gamma = {0}".format(g))
        plt.legend(handles=patches)
    plt.show()
    plt.clf()

print("Ridge MEAN ARRAY")
print(ridgeAccuracy_mean_array)

print("Ridge STD ARRAY")
print(ridgeAccuracy_std_array)