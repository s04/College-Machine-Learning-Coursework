import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data2.csv",comment='#')
# print(df.head())
X = np.array(df.iloc[:,0])
X = X.reshape(-1,1)
y = np.array(df.iloc[:,1])
y = y.reshape(-1,1)

norm = []

#normalising the data
ymin, ymax = min(y), max(y)
for i, val in enumerate(y):
    norm.append((val-ymin) / (ymax-ymin))

iter = 2000
class LinearRegGD:
    def __init__(self, learning_rate=0.000001, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
        self.m_losses = []
        self.b_losses = []
    
    def fit(self, X, y):
        m = 0
        b = 0
        n = 931
        print(n)
        
        for _ in range(self.iterations):
            m_gradient = (-2/n) * np.sum(X*(y - (m*X + b)))
            self.m_losses.append(m_gradient)
            b_gradient = (-2/n) * np.sum(y - (m*X + b)) 
            self.b_losses.append(b_gradient)
            b = b - (self.learning_rate*10000 * b_gradient)
            m = m - (self.learning_rate * m_gradient)
        self.m, self.b = m, b
    
    def losses(self):
        return self.m_losses, self.b_losses

    def predict(self, X):
        print("m = ", self.m, "b = ", self.b)
        return self.m*X + self.b


clf = LinearRegGD(iterations=iter)
clf.fit(X, norm)
m_l, b_l = clf.losses()

print("m loss", m_l[-5:-1])
print("b loss", b_l[-5:-1])

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.tight_layout(pad=2.0)
fig.suptitle("Linear Regression with Gradient Descent")
ax1.scatter(X, norm, color='green')
ax1.plot(X, clf.predict(X), color='black')
ax1.plot(X, -0.0005*X+1, color = "black")
ax1.set_ylabel('normalized data')
ax1.set_xlabel('number of points')
ax1.set_title('Data + Regression Line')
ax2.plot(range(1, iter+1), m_l)
ax2.set_ylabel('m-error')
ax2.set_xlabel('iterations')
ax2.set_title('m-loss function | lr = 0.000001')
ax3.plot(range(1, iter+1), b_l)
ax3.set_ylabel('b-error')
ax3.set_xlabel('iterations')
ax3.set_title('b-loss function | lr = 0.01')

plt.show()