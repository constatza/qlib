import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots

plt.style.use(['science', 'grid'])
plt.rcParams.update({
    'axes.grid': True,
    'axes.grid.which': 'both',
    'font.size': 23,
    'font.family': 'Serif',
    'figure.figsize': (7.5, 5.5),
    'figure.dpi': 500,
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


def compare(data, labels=None):
    num_experiments = data.shape[1]
    num_points = data.shape[0]

    fig, ax = plt.subplots()
    with sns.color_palette('Spectral'):
        sns.boxplot(data,
                    notch=True,
                    showfliers=False
                    ) 
    ax.set_xticklabels(labels)
    return fig

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

def plot_convergence(loss: object, title: object) -> object:
    """ Plot the loss function with respect to the number of iterations
        uses scientific format for the y-axis
     @rtype: object
     @param loss:
     @param title:
    """
    fig, ax = plt.subplots()
    ax.plot(loss)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(which='minor')
    return fig, ax

if __name__=='__main__':
    parent_dir = os.path.join(os.path.dirname(os.getcwd()), 'experiments', 'pauli')
    io_constant = ExperimentIO(parent_dir, 'q2-constant-custom-qc')
    io_knn = ExperimentIO(parent_dir, 'q2-nearby-custom-qc')

    vqls_params = io_constant.loadtxt('OptimalParameters')
    names = [f'$\\alpha_{i:d}$' for i in range(vqls_params.shape[1])]
    df = pd.DataFrame(vqls_params, columns=names)
    fig = pdf(df, dim=2,  vars=names[:3] )
    io_constant.savefig(fig, 'pairgrid')

    iterations_constant = io_knn.loadtxt('NumIterations', ndmin=2).astype(int)
    iterations_knn = io_constant.loadtxt('NumIterations', ndmin=2).astype(int)
    iterations = np.hstack([iterations_constant, iterations_knn])

    mean_knn = np.mean(iterations_knn)
    mean_constant = np.mean(iterations_constant)

    fig = compare(iterations, labels=['Original', 'Nearby'])
    io_constant.savefig(fig, 'compare-iterations')

    plt.show()

    print('EOF')
