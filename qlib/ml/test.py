#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

scaler = MinMaxScaler


input_path_vqls_parameters = "/home/archer/code/quantum/experiments/vqls8x8/results/critical/OptimalParameters.txt"
input_path_physical_parameters = "/home/archer/code/quantum/data/8x8/parameters.mat"

X = loadmat(input_path_physical_parameters)['parameterData'].T
y = np.loadtxt(input_path_vqls_parameters)

# X, y = make_regression(n_samples=1000, n_features=3, n_targets=19)
# y = np.atleast_2d(y)

y = np.hstack([np.cos(y), np.sin(y)])
yp = y
xp = X
for i in range(1, 1):
    yp = np.vstack([yp, y])
    xp = np.vstack([xp, i*2*np.pi + X])


    
X = xp
y = yp



X = scaler().fit_transform(X)
y = scaler().fit_transform(y)

# y = PCA(n_components=2).fit_transform(y)
df = pd.DataFrame(y[:, :5])
sns.pairplot(df, hue=2)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

model = Sequential()


model.add(Dense(5, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(10, activation='leaky_relu'))
model.add(Dense(15, activation='leaky_relu'))
model.add(Dense(30, activation='leaky_relu'))
model.add(Dense(y.shape[1], activation='linear'))

model.compile(loss='huber', optimizer='adam')

history = model.fit(x=X_train, y=y_train,
                    batch_size=50,
                    epochs=50, 
                    validation_split=0.3)

y_predicted = model.predict(X_test)

# pca = PCA(n_components=2).fit_transform(y)

loss = model.evaluate(X_test, y_test)



