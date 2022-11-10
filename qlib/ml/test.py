#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pandas as pd


plt.close('all')

scaler = MinMaxScaler


input_path_vqls_parameters = "/home/archer/code/quantum/experiments/vqls8x8/results/layers-2/OptimalParameters_2022-11-02_17-19.txt"
input_path_physical_parameters = "/home/archer/code/quantum/data/8x8/parameters.mat"
input_path_solutions = "/home/archer/code/quantum/experiments/vqls8x8/results/layers-2/Solutions_2022-11-02_17-19.txt"

y_raw = np.loadtxt(input_path_vqls_parameters)
X_raw = loadmat(input_path_physical_parameters)['parameterData'].T[:836]

# X_raw = np.loadtxt(input_path_solutions)
# y_raw = np.loadtxt(input_path_vqls_parameters)

# X, y = make_regression(n_samples=1000, n_features=3, n_targets=19)
# y = np.atleast_2d(y)

y = np.hstack([np.cos(4*y_raw), np.sin(4*y_raw)])
# y = y[:, 0:6]
# y = scaler().fit_transform(y_raw)
X = scaler().fit_transform(X_raw)
# y = scaler().fit_transform(y)
pca = PCA(n_components=3).fit(y)
y = pca.transform(y)

y_dim = y.shape[1]
X_dim = X.shape[1]

#######
# DRAW
#######




how_many = y_dim
df = pd.DataFrame(np.hstack([X, y]))
# df = pd.DataFrame(y)
sns.pairplot(df,
              hue=1,
              x_vars=np.arange(3),
              y_vars=np.arange(y_dim//2, y_dim//2+how_many))



fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], c = X[:, 1])
plt.title("simple 3D scatter plot")
plt.show()



fig, ax = plt.subplots()
ax.scatter(y[:, 0], y[:, 1], c = X[:, 1])
plt.show()


######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True
                                                    )



original_dims = X.shape[1] # 1000x3
output_dims = y.shape[1] # 1000x19


model = Sequential([
    Dense(5, input_shape=(original_dims,), activation='tanh'),
    Dense(5, activation='tanh'),
    Dense(5, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(output_dims, activation='linear'),
    ])


optimizer = Adam(learning_rate=1e-2,)





model.compile(loss='huber',
              optimizer=optimizer)

history = model.fit(x=X_train, y=y_train,
                    batch_size=20,
                    epochs=20,
                    validation_split=0.2)

y_predicted = model.predict(X_test)
loss = model.evaluate(X_test, y_test)
