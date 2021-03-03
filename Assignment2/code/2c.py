# id:19-19-19
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

df = pd.read_csv("data_etd.csv",header=None, comment="#", names=["X1","X2", "X12", "X22", "X3"])
X_all = df.drop(columns=["X3"])
X_labels = df["X3"]
X_train, X_test, y_train, y_test = train_test_split(X_all, X_labels, test_size=0.20, random_state=42)

#training the models
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
dummy_clf = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

params = clf.get_params()
print(params)

predictions = clf.predict(X_test)
dummy_predictions = dummy_clf.predict(X_test)

#splitting the data
X_predictions = X_test
X_predictions["X3"] = predictions

y_list_ok = y_test.tolist()

X_test = X_test.drop(columns=["X12"])
X_test = X_test.drop(columns=["X22"])
X_test = X_test.drop(columns=["X3"])

X_test_x1 = X_test["X1"].tolist()
X_test_x2 = X_test["X2"].tolist()

pred_tp_x1 = []
pred_tp_x2 = []

pred_tn_x1 = []
pred_tn_x2 = []

pred_fp_x1 = []
pred_fp_x2 = []

pred_fn_x1 = []
pred_fn_x2 = []

for i in range(len(y_list_ok)):
    if predictions[i] == y_list_ok[i]:
        if predictions[i] == 1:
            pred_tp_x1.append(X_test_x1[i])
            pred_tp_x2.append(X_test_x2[i])
        if predictions[i] == -1:
            pred_tn_x1.append(X_test_x1[i])
            pred_tn_x2.append(X_test_x2[i])
    if predictions[i] != y_list_ok[i]:
        if predictions[i] == 1:
            pred_fp_x1.append(X_test_x1[i])
            pred_fp_x2.append(X_test_x2[i])
        if predictions[i] == -1:
            pred_fn_x1.append(X_test_x1[i])
            pred_fn_x2.append(X_test_x2[i])

#end of data splitting

#performance evaluation
true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_list_ok, predictions).ravel()
conf_matrix = confusion_matrix(y_list_ok, predictions)
accuracy = accuracy_score(y_list_ok, predictions)

#plotting the data
print(true_neg, false_pos, false_neg, true_pos)
print("conf_matrix: \n", conf_matrix)
print("accuracy score: ", accuracy)

#performance evaluation
true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_list_ok, dummy_predictions).ravel()
conf_matrix = confusion_matrix(y_list_ok, dummy_predictions)
accuracy = accuracy_score(y_list_ok, dummy_predictions)

#plotting the data
print(true_neg, false_pos, false_neg, true_pos)
print("conf_matrix: \n", conf_matrix)
print("accuracy score: ", accuracy)

#plotting the data
plt.scatter(pred_tp_x1,pred_tp_x2, color='blue', label='Correct Positives')
plt.scatter(pred_tn_x1,pred_tn_x2, color='aqua', label='Correct Negatives')
plt.scatter(pred_fp_x1,pred_fp_x2, color='red', label='False Positives')
plt.scatter(pred_fn_x1,pred_fn_x2, color='lawngreen', label='False Negatives')
plt.legend(loc='upper left', shadow=True)
plt.title("Predicted Model Values")
plt.xlabel("x1_pred")
plt.ylabel("x2_pred")

plt.show()