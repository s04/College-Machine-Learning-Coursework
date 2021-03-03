import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv",comment='#')
# print(df.head())
X = np.array(df.iloc[:,0])
X = X.reshape(-1,1)
y = np.array(df.iloc[:,1])
y = y.reshape(-1,1)

norm = []

# s = sum(y); norm = [float(i)/s for i in y]
# norm2 = [((float(i)-min(y))/max(y)-min(y)) for i in y]

ymin, ymax = min(y), max(y)
for i, val in enumerate(y):
    norm.append((val-ymin) / (ymax-ymin))

# norm_fix = []

# for i in range(len(norm)):
#     norm_fix.append(norm[i][0])

# Xfix = []

# for i in range(len(X)):
#     Xfix.append(X[i][0])

# X = Xfix

# for i in range(len(norm)):
#     norm[i][0] = norm[i][0]*10
# print(norm)

# plt.plot(X, y)
# plt.show()

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.0000001, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
    
    def fit(self, X, y):
        m = 0
        b = 0
        n = 931
        print(n)
        # print(y[0:10])
        for _ in range(self.iterations):
            Y_pred = m*X + b
            m_gradient = (-2/n) * np.sum(X*(y - (Y_pred)))
            b_gradient = (-2/n) * np.sum(y - (Y_pred)) 
            b = b + (self.learning_rate * b_gradient)
            m = m - (self.learning_rate * m_gradient)

            #print("b = ",b, "m = ", m)
            #print("b^ = ",b_gradient,"m^ = ", m_gradient)
        self.m, self.b = m, b
        
    def predict(self, X):
        print("m = ", self.m, "b = ", self.b)
        return self.m*X + self.b


clf = GradientDescentLinearRegression()
clf.fit(X, norm)
# clf.predict(X)
plt.style.use('fivethirtyeight')

plt.scatter(X, norm, color='black')
plt.plot(X, clf.predict(X))
plt.gca().set_title("Gradient Descent Linear Regressor")

plt.show()


# m = 0
# c = 0
# L = 0.0001  # The learning Rate
# epochs = 4  # The number of iterations to perform gradient descent

# n = 931 # Number of elements in X

# # Performing Gradient Descent 
# for i in range(epochs): 
#     Y_pred = m*X + c  # The current predicted value of Y
#     D_m = (-2/n) * sum(X * (norm - Y_pred))  # Derivative wrt m
#     D_c = (-2/n) * sum(norm - Y_pred)  # Derivative wrt c
#     m = m - (L * D_m)  # Update m
#     c = c - (L * D_c)  # Update c
    
# # print (m, c)

# # plt.scatter(X, y)

# # plt.show()

# # Making predictions
# Y_pred = m*X + c

# plt.scatter(X, norm) 
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
# plt.show()