#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/ml_scratch/linear_model.linear_regression.py was created on 2019/03/04.
file in :
Author: Charles_Lai
Email: lai.bluejay@gmail.com



"""
import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/.." % root)

from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np
import math
from base_model import Model

class L1Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        # 求范式 vector/ matrix norms
        return self.alpha * np.linalg.norm(w)
    
    def grad(self, w):
        return self.alpha * np.sign(w)

class L2Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
    
    def grad(self, w):
        return self.alpha * w

class ElasticNet(object):
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contruction = self.l1_ratio * self.alpha * np.linalg.norm(w)
        l2_contruction = (1-self.l1_ratio) * self.alpha * 0.5 * w.T.dot(w)
        return l1_contruction + l2_contruction
    
    def grad(self, w):
        return self.l1_ratio * self.alpha * np.sign(w) + (1 - self.l1_ratio) * self.alpha * w


    


class LinearRegression(Model):
    """[LinearRegression]
    1. init weights
    2. cost function : J(x) = \frac{1}{2n}\sum((\hat(y) - y)^2)
        loss :l2 loss
    3. gradient : d = \frac{1}{n} (\hat(y) - y )^2
    Arguments:
        Model {[type]} -- [description]
    """

    def __init__(self, learning_rate=0.00001, iterations=100000, regularization=None):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.iterarions = iterations
        if regularization:
            self.regularization = regularization
        else:
            # no regularization
            self.regularization = lambda x:0
            self.regularization.grad = lambda x:0
    
    def l2_loss(self, y_pred, y):
        r = self.regularization(self.theta)
        dd = np.mean(0.5 * (y-y_pred)**2 + r)
        self.mse = np.mean(0.5 * (y - y_pred)**2 + r)
        return self.mse

    def gradient_descent(self, X, y, y_hat):
        grad = (-np.dot((y-y_hat), X))/(self.n_samples) + self.regularization.grad(self.theta)
        self.theta = self.theta - self.learning_rate * grad


    def init_weights(self, X):
        self.n_samples , self.n_features = X.shape
        limit = 1.0/math.sqrt(self.n_features)
        self.theta = np.random.uniform(-limit, limit, self.n_features)
        bias = 1
        self.theta = np.insert(self.theta, 0, bias)

    def add_bias_col(self, X):
        n_samples = X.shape[0]
        bias_col = np.ones((n_samples, 1))
        X = np.concatenate([bias_col, X], axis=1)
        return X

    def fit(self, X, y):
        self.init_weights(X)
        X = self.add_bias_col(X)
        self.training_errors = list()
        print(self.theta)
        for iter in range(self.iterarions):
            y_pred = X.dot(self.theta)
            # Calculate l2 loss
            mse = self.l2_loss(y_pred, y)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.theta))
            self.training_errors.append(mse)
            # # Gradient of l2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.theta)
            # # Update the weights
            self.theta -= self.learning_rate * grad_w
            # self.gradient_descent(X, y , y_pred)
            # print(self.theta, grad_w)
            if iter % 100 == 0: print("{0}th iter mse: {1}".format(iter, mse))

    def predict(self, X):
        X = self.add_bias_col(X)
        y_hat = np.dot(X, self.theta)
        return y_hat
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

def main():

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    print(reg.score(X,y ))
    print(reg.predict(np.array([[3, 5]])))
    sys.exit(0)



    X, y = make_regression(n_samples=100000, n_features=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    diabetes = datasets.load_diabetes()


    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    X_train = diabetes_X[:-20]
    X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]

    # 可自行设置模型参数，如正则化，梯度下降轮数学习率等

    model = LinearRegression(learning_rate=0.1,iterations=10, regularization=None)

    model.fit(X_train,y_train)

    # Training error plot 画loss的图
    # n = len(model.training_errors)
    # training, = plt.plot(range(n), model.training_errors, label="Training Error")
    # plt.legend(handles=[training])
    # plt.title("Error Plot")
    # plt.ylabel('Mean Squared Error')
    # plt.xlabel('Iterations')
    # plt.savefig('./train.png')

    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: %s" % (mse))

    y_pred_line = model.predict(X_test)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results，画拟合情况的图
    # m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    # m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    # # plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    # plt.suptitle("Linear Regression")
    # plt.title("MSE: %.2f" % mse, fontsize=10)
    # plt.xlabel('Day')
    # plt.ylabel('Temperature in Celcius')
    # plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    # plt.savefig("linear_regression.png")
    
    ax = plt.plot()
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred_line, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.savefig('linear_fit.png')

if __name__ == "__main__":
    main()



    
