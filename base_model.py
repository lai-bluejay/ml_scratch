#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/ml_scratch.base_model.py was created on 2019/03/04.
file in :
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

class Model(object):

    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError