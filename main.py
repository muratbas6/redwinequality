import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

my_data = pd.read_csv('data.txt',names=["x","x1","x2","x3","x4","y"]) #read the data
z

X = my_data.iloc[:,0:4]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y = my_data.iloc[:,4:5].values
theta = np.zeros([1,5])
alpha = 0.01
iters = 10000



def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta, cost



g, cost = gradientDescent(X, y, theta, iters, alpha)
print(g)

finalCost = computeCost(X, y, g)





fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

predict = np.array([1,5.0,3.4,1.6,0.4])
print(predict @ g.T)
plt.show()
