#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/ml_scratch/linear_model.logistic_regression.py was created on 2019/03/04.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report

import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/.." % root)
sys.path.append(u"{0:s}".format(root))

from utils import Plot
from base_model import Model

"""[summary]

Hypothesis function $H_\theta = \frac{1}{1+exp(-z)} = \frac{1}{1+exp(-\theta x)} = \hat{y}$ .
cross_entropy: 
$$
J(x) = \frac{1}{-n} \sum (y*log(\hat{y})+(1-y)log(1-\hat{y}) )
$$

the gradient of J(x):
$$
    \frac{d}{d\theta}J(x) = -\frac{1}{2n}\sum{x(y-\hat(y))}
$$
update $\theta = \theta - \alpha \frac{dx}{dJ}$

Gradient descent for Logistic Regression is again performed in the same way:

<ul>
	<li>Make an initial but intelligent guess for the values of the parameters $ \theta $.</li>
	<li>Keep iterating while the value of the cost function has not met your criteria:
<ul>
	<li>With the current values of $ \theta $, calculate the gradient of the cost function J  ( $ \Delta \theta = - \alpha \frac{d}{d\theta} J(x)$ ).</li>
	<li>Update the values for the parameters $ \theta := \theta + \alpha \Delta \theta $</li>
	<li>Fill in these new values in the hypothesis function and calculate again the value of the cost function;</li>
</ul>
</li>
</ul>
Returns:
    [type] -- [description]
"""



class LogisticRegression(Model):
    """
    parameters:
    迭代次数、学习率、初始化
    """

    def __init__(self, learnging_rate=0.1, iteration=1000):
        # super(CLASS_NAME, self).__init__(*args, **kwargs)
        self.learning_rate = learnging_rate
        self.iteration = iteration
        self._b = 0
        self.theta = 0
        self.n_samples = 0
        self.n_features = 0
        self._y_hat = 0
        pass

    def _initiallize_weights(self, n_features):
        limit = 1.0 / math.sqrt(self.n_features)
        w = np.random.uniform(-limit, limit, (n_features, ))
        b = 1
        self.theta = np.insert(w, 0, b, axis=0)

    def add_bias_col(self, X):
        n_samples = X.shape[0]
        bias_col = np.ones((n_samples, 1))
        return np.concatenate([bias_col, X], axis=1)

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self._initiallize_weights(self.n_features)
        X = self.add_bias_col(X)
        # y = np.reshape(y, )
        for iter in range(self.iteration):
            cost = self.gradient_descent(X, y)
            if iter % 100 == 0: print("iteration %s : cost %s " % (iter, cost))

    def cost_function(self, y_hat, y):
        """cross-entrophy
        remove divide 0
        """
        cost = (-1.0/self.n_samples) *\
            np.sum(y * np.log(y_hat+0.0000001) + (1-y)*np.log(1-y_hat+0.0000001))
        return cost

    def gradient_descent(self, X, y):
        '''
        Updates the weights by computing: W - learning rate * gradient.
        gradient = x * (sigmoid(z) - y)
        Returns:
        W - Updates weights
        '''
        self._y_hat = self.sigmoid(X)
        grad = np.dot(X.T, (self._y_hat - y))
        cost = self.cost_function(self._y_hat, y)
        print(self.theta, grad)
        self.theta = self.theta - self.learning_rate * grad
        # self._w = self._w - grad
        return cost

    def optimized_method(self):
        pass

    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-1.0 * np.dot(X, self.theta)))

    def predict(self, X):
        X = self.add_bias_col(X)
        y_pred = self.sigmoid(X)
        return y_pred
    
    def predict_label(self, X, threshold=0.5):
        X = self.add_bias_col(X)
        y_hat = self.sigmoid(X)
        y_pred = list()
        for y in y_hat:
            if y > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)


def main():
    # Load dataset
    data = datasets.load_iris()
    # X = normalize(data.data[data.target != 0])
    X = data.data[data.target != 0]
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)

    clf = LogisticRegression(iteration=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = roc_auc_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    y_pred = clf.predict_label(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()