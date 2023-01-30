#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:18:48 2022

@author: archer
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD, RMSprop, Adam
# from tensorflow.keras.layers import Normalization
# from qlib.ml.utils import SinCosTransformation

import matplotlib.pyplot as plt
# import scienceplots


# plt.style.use(['science', 'nature'])

experiments_dir = os.path.join("output", "q2_random")

input_path_vqls_parameters = os.path.join(experiments_dir, "OptimalParameters")
input_path_physical_parameters = os.path.join("input", "parameters-num_qubits_2.npy")

y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = np.load(input_path_physical_parameters)


X = X_raw
y = y_raw


y_dim = y.shape[1]
X_dim = X.shape[1]


######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    shuffle=True
                                                    )

# scalerX = Normalization()
# scalerX.adapt(X_train)


original_dims = X_train.shape[1] # 1000x3
output_dims = y_train.shape[1] # 1000x19


layer_size = 10 

model = Sequential([Input(shape=(original_dims,)),
                  #   scalerX,
                    Dense(layer_size,
                          input_shape=(original_dims,),
                          activation='tanh'),
                    Dense(layer_size, activation='tanh'),
                    Dense(output_dims, activation='linear'),
    ])

optimizer = Adam(learning_rate=0.1)

model.compile(loss='mse',
              optimizer=optimizer)


history = model.fit(x=X_train, y=y_train,
                    batch_size=50,
                    epochs=500,
                    validation_split=0.3)

loss = model.evaluate(X_test, y_test)

y_predicted = model(X_test)


fig, axes = plt.subplots(nrows=3, ncols=3, 
                        figsize=(10, 10))
fig.suptitle(f'MSE loss = {loss:.2f}')
for i in range(3):
      for j in range(3):
            ax = axes[i, j]

            xx = X_test[:, j]
            yp = y_predicted[:, i + 3]
            yt = y_test[:, i]


            ax.plot(xx, yt, 'o', label='test')
            ax.plot(xx, yp, 'r+', label='predicted')
            ax.set_xlabel(f'$x_{j}$')
            ax.set_ylabel(f'$y_{i}$')
            ax.grid()
            if i==0 and j==0:
                  ax.legend()




plt.figure()
plt.scatter(y[:, 0], y[:, 1])

plt.show()
# fig.savefig(os.path.join('output'/img/mlp.png', dpi=400)
model.save(os.path.join('models','mlp2'))

