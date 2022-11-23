#!/usr/bin/env python

import os
import numpy as np

from sklearn.decomposition import PCA
from scipy.io import loadmat


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pandas as pd
# import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])

plt.close('all')

experiments_dir = "/home/archer/code/quantum/experiments/custom4x4/results/2022-11-17_21-09"

input_path_vqls_parameters = os.path.join(experiments_dir, "OptimalParameters")
input_path_physical_parameters = os.path.join(experiments_dir, "parameters.in")

y = np.loadtxt(input_path_vqls_parameters)
X = np.loadtxt(input_path_physical_parameters)

# freq = 0.5
# y = np.hstack([np.cos(freq*y), np.sin(freq*y)])

# pca = PCA(n_components=3).fit(y)
# y = pca.transform(y)

ydim = y.shape[1]
xdim = X.shape[1]

#######
# DRAW
#######


how_many = ydim
df = pd.DataFrame(np.hstack([X, y]))
# df = pd.DataFrame(y)
# sns.pairplot(df,
#              x_vars = [0, 1, 4], y_vars=[2, 3, 4], diag_kind='hist')



fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], c=X[:, 1])
plt.title("simple 3D scatter plot")
plt.show()



fig, ax = plt.subplots()
ax.scatter(X[:, 0], y[:, 2], c = X[:, 1])
plt.show()


size = (30, 30)
xx1 = X[:, 0].reshape(size)
xx2 = X[:, 1].reshape(size)
yy = y[:, 2].reshape(size)

fig, ax = plt.subplots()
plot = ax.contourf(xx1, xx2, yy)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
fig.colorbar(plot)

fig.savefig(os.path.join(experiments_dir,'img/angle3-vs-x1_x2.png'), dpi=400)
