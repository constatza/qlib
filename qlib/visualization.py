import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots

plt.style.use(['science', 'nature'])
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


def compare(data, title=''):
    num_experiments = data.shape[1]
    num_points = data.shape[0]

    fig, axes = plt.subplots(num_experiments, sharex=True)
    for i in range(num_experiments):
        ax = axes[i]
        series = data[:, i]
        total = series.sum()
        ax.hist(series, bins=10, density=True)
        ax.set_title(f'Total {title:s}: {total}')
    return fig

if __name__=='__main__':
    parent_dir = os.path.join(os.path.dirname(os.getcwd()), 'experiments', 'pauli')
    filename = 'OptimalParameters'
    figname = 'grid'
    io_knn = ExperimentIO(parent_dir, 'q2_knn')
    io_mlp = ExperimentIO(parent_dir, 'q2_random')

    vqls_params = io_knn.loadtxt(filename)
    names = [f'$\\alpha_{i:d}$' for i in range(vqls_params.shape[1])]
    df = pd.DataFrame(vqls_params, columns=names)
    fig = pdf(df, dim=2,  vars=names[4:7] )
    io_knn.savefig(fig, figname)

    iterations_mlp = io_mlp.loadtxt('NumFunctionEvaluations', ndmin=2).astype(int)[:900]
    iterations_knn = io_knn.loadtxt('NumFunctionEvaluations', ndmin=2).astype(int)[:900]
    iterations = np.hstack([iterations_knn, iterations_mlp])

    mean_knn = np.mean(iterations_knn)
    mean_mlp = np.mean(iterations_mlp)

    fig = compare(iterations, 'iterations')
    io_knn.savefig(fig, 'iterations')

    plt.show()

    print('EOF')
