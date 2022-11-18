#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:48:38 2022

@author: archer
"""

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
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.layers import Normalization
from qlib.ml.utils import SinCosTransformation
import matplotlib.pyplot as plt
import os

experiments_dir = "/home/archer/code/quantum/experiments/custom4x4/results/2022-11-17_21-09"


input_path_vqls_parameters = os.path.join(experiments_dir, "OptimalParameters")
input_path_physical_parameters = os.path.join(experiments_dir, "parameters.in")


y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = np.loadtxt(input_path_physical_parameters)


scale=1



X = X_raw
y = y_raw


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


original_dims = X_train.shape[1] # 1000x3
output_dims = y_train.shape[1] # 1000x19


layer_size = 3

model = Sequential([Input(shape=(original_dims,)),
                    scalerX,
                    Dense(layer_size,
                          input_shape=(original_dims,),
                          activation='tanh'),
                    # Dense(layer_size, activation='tanh'),
                    Dense(output_dims, activation='linear'),
    ])






optimizer = Adam(learning_rate=1e-2,)



model.compile(loss='huber',
              optimizer=optimizer)



history = model.fit(x=X_train, y=y_train,
                    batch_size=5,
                    epochs=50,
                    validation_split=0.3)

loss = model.evaluate(X_test, y_test)



y_predicted = model.predict(X_test)


xx = X_test[:, 1]
yp = y_predicted[:, 2]
yt = y_test[:, 2]



fig, ax = plt.subplots()
ax.plot(xx, yt, 'o', label='test')
ax.plot(xx, yp, 'r+', label='predicted')
plt.legend()




model.pop()
y_predicted = model.predict(X_test)

# plt.plot(xx, y_predicted[:, 2], 'o',
#           X_raw[:, 1], y_raw[:, 2], 'o')

model.save('model0')
