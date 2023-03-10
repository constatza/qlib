#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:18:48 2022

@author: archer
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.layers import Normalization
from tensorflow.keras import regularizers
from qlib.ml.utils import SinCosTransformation
import matplotlib.pyplot as plt
import os

experiments_dir = "./results/2022-11-17_21-09"


input_path_vqls_parameters = os.path.join(experiments_dir, "OptimalParameters")
input_path_physical_parameters = os.path.join(experiments_dir, "parameters.in")


y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = np.loadtxt(input_path_physical_parameters)


scale=1


# X = scalerX().fit_transform(X_raw)
X = X_raw
y = y_raw
scalerY = SinCosTransformation(scale=scale)
# y = scalerY(y_raw).numpy()



# pca = PCA(n_components=3).fit(y)
# y = pca.transform(y)

y_dim = y.shape[1]
X_dim = X.shape[1]



######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.8,
                                                    shuffle=True,
                                                    )

scalerX = Normalization()
scalerX.adapt(X_train)
y_train = scalerY(y_train).numpy()

original_dims = X_train.shape[1] # 1000x3
output_dims = y_train.shape[1] # 1000x19


layer_size = 15

model = Sequential([Input(shape=(original_dims,)),
                    scalerX,
                    Dense(layer_size,
                          input_shape=(original_dims,),
                          activation='tanh'),
                    # Dense(layer_size, activation='tanh'),
                    Dense(output_dims//2, activation='linear'),
                    scalerY,
    ])






optimizer = Adam(learning_rate=1e-2,)



model.compile(loss='huber',
              optimizer=optimizer)



history = model.fit(x=X_train, y=y_train,
                    batch_size=10,
                    epochs=500,
                    validation_split=0.3)


scalerY.inverse = True
model.add(scalerY)
loss = model.evaluate(X_test, y_test)



y_predicted = model.predict(X_test)

step=1
xx = X_test[::step, 1]
yp = y_predicted[::step, 2]
yt = y_test[::step, 2]



fig, ax = plt.subplots()
ax.plot(xx, yt, 'o', label='test')
ax.plot(xx, yp, 'r+', label='predicted')
plt.legend()





y_predicted = model.predict(X_test)

# plt.plot(xx, y_predicted[:, 2], 'o',
#           X_raw[:, 1], y_raw[:, 2], 'o')

model.save('model0')

size = (30, 30)
xx1 = X[:, 0].reshape(size)
xx2 = X[:, 1].reshape(size)
yy = y[:, 0].reshape(size)

fig, ax = plt.subplots()
plot = ax.contourf(xx1, xx2, yy)
# ax.plot(xx, yp, 'r+', label='predicted')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
fig.colorbar(plot)

fig.savefig('untrainable.png', dpi=400)
