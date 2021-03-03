# id:19-19-19
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv("data.csv",header=None, comment="#", names=["X1","X2","X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]

X_train, X_test, y_train, y_test = train_test_split(X_all, X_labels, test_size=0.33, random_state=42)

X_predictions = X_test

C_array = [1, 100, 1000, 10000, 0.0001, 0.00001]

fig, axs = plt.subplots(2,3,sharex=True, sharey=True)

fig.suptitle("Data + Predicted Data + Decision boundary\n Based on different C kernels")

axs = axs.flatten()

#splitting the data
#Seperating 1 and -1 values into Xpositive and Xnegative
Xp = df.loc[df["X3"]==1]
Xm = df.loc[df["X3"]==-1]
Xp_labels = Xp["X3"]
Xm_labels = Xm["X3"]

print(type(axs))

#training the models
def svm_fit_test_measure(c_val,i):
    #training the models
    clf = LinearSVC(C=c_val)
    clf.fit(X_train, y_train)
    X_predictions = X_test
    predicted_y = clf.predict(X_test[["X1","X2"]])

    X_predictions["X3"] = predicted_y

    Xp_predictions = X_predictions.loc[X_predictions["X3"]==1]
    Xm_predictions = X_predictions.loc[X_predictions["X3"]==-1]

    y_list_ok = y_test.tolist()

    #performance evaluation
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_list_ok, predicted_y).ravel()
    conf_matrix = confusion_matrix(y_list_ok, predicted_y)
    accuracy = accuracy_score(y_list_ok, predicted_y)

    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T

    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    xmin, xmax = -1, 1
    xd = np.array([xmin, xmax])
    yd = m*xd + c

    
    #plotting the data
    axs[i].plot(xd, yd, 'k', lw=3, color='g')
    axs[i].set_xlim([-1,1])
    axs[i].set_ylim([-1,1])
    axs[i].set_title("C = " + str(c_val))
    axs[i].scatter(Xp["X1"],Xp["X2"], color='b', s = 7)
    axs[i].scatter(Xm["X1"],Xm["X2"], color='aqua', s = 7)
    axs[i].scatter(Xp_predictions["X1"], Xp_predictions["X2"], color='hotpink', s = 10)
    axs[i].scatter(Xm_predictions["X1"], Xm_predictions["X2"], color='r', s = 10)

    print(true_neg, false_pos, false_neg, true_pos)
    print("conf_matrix: \n", conf_matrix)
    print("accuracy score: ", accuracy)

    return [c_val, true_neg, false_pos, false_neg, true_pos, accuracy, clf.get_params()]

svm_array = []

for i in range(len(C_array)):
    svm_array.append(svm_fit_test_measure(C_array[i],i))

plt.show()
print(svm_array)
