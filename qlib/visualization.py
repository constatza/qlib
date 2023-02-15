import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import scienceplots

# plt.style.use(['science', 'nature'])
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'Serif',
    'axes.grid': True
})

class ExperimentIO:

    def __init__(self, maindir, subdir, input=False):
        self.maindir = maindir
        if input:
            data = 'input'
        else:
            data = 'output' 
        self.data = os.path.join(maindir, data) 
        self.subdir = os.path.join(self.data, subdir)

    def loadtxt(self, filename, **kwargs):
        filepath = self.subdir_join(filename)
        return np.loadtxt(filepath, **kwargs)
    
    def savefig(self, fig, name, ext='.png'):
        try:
            os.makedirs(self.subdir_join('img'))
        except:
            pass
        savepath = self.subdir_join('img', name + ext)
        fig.savefig(savepath, dpi=500)
    
    def subdir_join(self, *other):
        return os.path.join(self.subdir, *other)


def pdf(df, dim=1, **kwargs):

    if dim==2:
        g = sns.pairplot(df, **kwargs)
        g.map_lower(sns.kdeplot)
    else:
        fig, axes = plt.subplots(3,1)
    
    return plt.gcf()


def compare(data, names, title=''):
    num_experiments = data.shape[1]
    num_points = data.shape[0]

    fig, ax = plt.subplots()
    sns.boxplot(data, width=0.5, notch=True, showfliers=False) 
    ax.set_xticks([0, 1, 2], labels=names)
    ax.set_title(title)
    return fig, ax

def pair_grid(data, title=''):
    num_series = data.shape[1]
    fig, axes = plt.subplots(nrows=num_series, ncols=num_series, 
                        figsize=(10, 10))
    fig.suptitle(title)
    for i in range(num_series):
        for j in range(num_series):
                ax = axes[i, j]

                xx = data[:, j]
                yp = data[:, i]

                ax.scatter(xx, yp)
                ax.set_xlabel(f'$x_{j}$')
                ax.set_ylabel(f'$y_{i}$')
                ax.grid()
                if i==0 and j==0:
                    ax.legend()
    return fig, axes

if __name__=='__main__':
    parent_dir = os.path.join(os.path.dirname(os.getcwd()), 'experiments', 'pauli')

    io_constant = ExperimentIO(parent_dir, 'q3-original')
    io_mlp = ExperimentIO(parent_dir, 'q3_nn')
    io_knn = ExperimentIO(parent_dir, 'q3_last_best')

    size = 760
    iterations_constant = io_constant.loadtxt('NumIterations', ndmin=2).astype(int)[:size]
    iterations_knn = io_knn.loadtxt('NumIterations', ndmin=2).astype(int)[:size]
    iterations_mlp = io_mlp.loadtxt('NumIterations', ndmin=2).astype(int)[:size]
    iterations = np.hstack([iterations_constant, iterations_knn, iterations_mlp])

    mean_knn = np.mean(iterations_knn)
    mean_constant = np.mean(iterations_constant)

    fig, ax = compare(iterations, names=['Constant', 'Nearby', 'MLP'], title='8x8 system - 3 qubits')
    ax.set_ylabel('Iterations')
    io_constant.savefig(fig, 'iterations')

    plt.show()

    print('EOF')
