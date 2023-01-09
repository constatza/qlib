import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots

plt.style.use(['science', ])
plt.rcParams.update({
    'font.size': 20,
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

    def loadtxt(self, filename):
        filepath = self.subdir_join(filename)
        return np.loadtxt(filepath)
    
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

    # fig, axes = plt.subplots(3,1)
    if dim==2:
        g = sns.pairplot(df, **kwargs)
        g.map_lower(sns.kdeplot)
        return plt.gcf()



if __name__=='__main__':
    parent_dir = os.path.join(os.path.dirname(os.getcwd()), 'experiments', 'pauli')
    filename = 'OptimalParameters'
    figname = 'pairgrid'
    io = ExperimentIO(parent_dir, 'num-qubits-3_2022-12-21_11-46')

    vqls_params = io.loadtxt(filename)
    names = [f'$\\alpha_{i:d}$' for i in range(vqls_params.shape[1])]
    df = pd.DataFrame(vqls_params, columns=names)
    fig = pdf(df, dim=2,  vars=names[4:7] )
    plt.show()
    io.savefig(fig, figname)
    

    print('EOF')
