#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:18:48 2022

@author: archer
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat
from tensorflow.keras.layers import Normalization
from qlib.ml.utils import KNNRegressor
import matplotlib.pyplot as plt
import os

experiments_dir = os.path.join("output", "num-qubits-3_2022-12-21_11-46")

input_path_vqls_parameters = os.path.join(experiments_dir, "OptimalParameters")
input_path_physical_parameters = os.path.join("input", "parameters-num_qubits_3.npy")

y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = np.load(input_path_physical_parameters)

scale=1

X = X_raw
y = y_raw


y_dim = y.shape[1]
X_dim = X.shape[1]


######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.7,
                                                    )

scalerX = Normalization()
scalerX.adapt(X_train)


original_dims = X_train.shape[1] # 1000x3
output_dims = y_train.shape[1] # 1000x19

import tensorflow as tf

# Initialize model
model = KNNRegressor(k=3, )


# Store data in model's weight variable
model.fit(X_train, y_train)
model.compile(loss='mse')

# Make predictions on synthetic data
predictions = model(X_test)
loss = model.evaluate(X_test, y_test)

y_predicted = model(X_test)

xx = X_test[:, 1]
yp = y_predicted[:, 2]
yt = y_test[:, 2]


fig, ax = plt.subplots()
ax.plot(xx, yt, 'o', label='test')
ax.plot(xx, yp, 'r+', label='predicted')
plt.legend()


# plt.plot(xx, y_predicted[:, 2], 'o',
#           X_raw[:, 1], y_raw[:, 2], 'o')

plt.show()
model.save('./models/knn')
