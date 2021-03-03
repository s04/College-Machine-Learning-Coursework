import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
plt.rc('text', usetex=True)

pts = np.loadtxt('linpts.txt')
X = pts[:,:2]
Y = pts[:,2].astype('int')

print(X)
print("---")
print(Y)
# # Fit the data to a logistic regression model.
# clf = sklearn.linear_model.LogisticRegression()
# clf.fit(X, Y)

# # Retrieve the model parameters.
# b = clf.intercept_[0]
# w1, w2 = clf.coef_.T
# # Calculate the intercept and gradient of the decision boundary.
# c = -b/w2
# m = -w1/w2

# # Plot the data and the classification with the decision boundary.
# xmin, xmax = -1, 2
# ymin, ymax = -1, 2.5
# xd = np.array([xmin, xmax])
# yd = m*xd + c
# plt.plot(xd, yd, 'k', lw=1, ls='--')
# # plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
# # plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

# plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
# plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
# plt.ylabel(r'$x_2$')
# plt.xlabel(r'$x_1$')
# plt.show()