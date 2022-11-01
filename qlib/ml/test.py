#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pandas as pd

scaler = MinMaxScaler


input_path_vqls_parameters = "/home/archer/code/quantum/experiments/vqls8x8/results/critical/OptimalParameters.txt"
input_path_physical_parameters = "/home/archer/code/quantum/data/8x8/parameters.mat"
input_path_solutions = "/home/archer/code/quantum/experiments/vqls8x8/results/critical/Solutions.txt"

# y_raw = np.loadtxt(input_path_solutions)
# X_raw = loadmat(input_path_physical_parameters)['parameterData'].T

X_raw = np.loadtxt(input_path_solutions)
y_raw = np.loadtxt(input_path_vqls_parameters)

# X, y = make_regression(n_samples=1000, n_features=3, n_targets=19)
# y = np.atleast_2d(y)



#######
# DRAW 
#######
y = np.hstack([np.cos(y_raw), np.sin(y_raw)])
# y = scaler().fit_transform(y_raw)
X = scaler().fit_transform(X_raw)
# y = scaler().fit_transform(y)
# pca = PCA(n_components=3).fit(y)
# y = pca.transform(y)

# how_many = 5
# y_dim = y.shape[1]
# X_dim = X.shape[1]
# # df = pd.DataFrame(np.hstack([X, y[:, -8:-1]]))
# df = pd.DataFrame(y)
# sns.pairplot(df, 
#              hue=0, 
#              x_vars=np.arange(how_many),
#              y_vars=np.arange(y_dim//2, y_dim//2+how_many))
# plt.show()

 
# Creating figure
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# # Creating plot
# ax.scatter3D(y[:, 0], y[:, 1], y[:, 9], c = X[:,1])
# plt.title("simple 3D scatter plot")
 
# # show plot
# plt.show()


######
# TRAIN
######

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    shuffle=True
                                                    )



original_dims = y.shape[1]
latent_dims = X.shape[1]

encoder = Sequential([
    Dense(15, input_shape=(original_dims,), activation='relu'),
    Dense(latent_dims, activation='relu'),
    ])  



decoder = Sequential([
    Dense(15, input_shape=(latent_dims, ), activation='relu'),
    Dense(original_dims, activation='tanh')
    ])  


original_input = Input(shape=(original_dims,))
latent_vector = encoder(original_input)
original_output = decoder(latent_vector)
autoencoder = Model(inputs=original_input, outputs=original_output)

autoencoder.compile(loss='mse', 
              optimizer='nadam')

history = autoencoder.fit(x=y_train, y=y_train,
                    batch_size=30,
                    epochs=200, 
                    validation_split=0.3)

y_predicted = encoder.predict(X_test)

# pca = PCA(n_components=2).fit_transform(y)

loss = encoder.evaluate(X_test, y_test)



