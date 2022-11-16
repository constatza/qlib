#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:18:48 2022

@author: archer
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD, RMSprop, Adam
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.layers import Normalization
from tensorflow.keras import regularizers
from utils import AngleScaler, SinCosTransformation
import matplotlib.pyplot as plt




input_path_vqls_parameters = "/home/archer/code/quantum/experiments/vqls8x8/results/continuous/OptimalParameters_2022-11-02_17-19.txt"
input_path_physical_parameters = "/home/archer/code/quantum/data/8x8/parameters.mat"
input_path_solutions = "/home/archer/code/quantum/experiments/vqls8x8/results/layers-2/Solutions_2022-11-02_17-19.txt"

y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = loadmat(input_path_physical_parameters)['parameterData'].T[:-2]

# X_raw = np.loadtxt(input_path_solutions)
# y_raw = np.loadtxt(input_path_vqls_parameters)

scale=1/2


# X = scalerX().fit_transform(X_raw)
X = X_raw
scalerY = SinCosTransformation(scale=scale)
y = scalerY(y_raw).numpy()



# pca = PCA(n_components=3).fit(y)
# y = pca.transform(y)

y_dim = y.shape[1]
X_dim = X.shape[1]



######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    )

scalerX = Normalization()
scalerX.adapt(X_train)


original_dims = X_train.shape[1] # 1000x3
output_dims = y_train.shape[1] # 1000x19


layer_size = 20

model = Sequential([Input(shape=(original_dims,)),
                    scalerX,
                    Dense(layer_size,
                          input_shape=(original_dims,),
                          activation='tanh'),
                    Dense(layer_size, activation='tanh'),
                    Dense(output_dims//2, activation='linear',
                          activity_regularizer=regularizers.L2(1e-12)),
                    scalerY,
    ])


optimizer = Adam(learning_rate=1e-2,)



model.compile(loss='huber',
              optimizer=optimizer)



history = model.fit(x=X_train, y=y_train,
                    batch_size=10,
                    epochs=50,
                    validation_split=0.3)

loss = model.evaluate(X_test, y_test)



y_predicted = model.predict(X_test)

step=5
xx = X_test[::step, 1]
yp = y_predicted[::step, 2]
yt = y_test[::step, 2]



fig, ax = plt.subplots()
ax.plot(xx, yt, 'o', label='test')
ax.plot(xx, yp, 'r+', label='predicted')
plt.legend()




model.pop()
y_predicted = model.predict(X_test)

# plt.plot(xx, y_predicted[:, 2], 'o',
#           X_raw[:, 1], y_raw[:, 2], 'o')

model.save('model0')
