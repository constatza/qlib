#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 29 15:38:15 2022

@author: archer
"""

import numpy as np
from scipy.optimize import Bounds
from qiskit import Aer
from scipy.io import loadmat
from numpy.linalg import det, cond
from qlib.solvers.vqls import VQLS, FixedAnsatz, Experiment
from qlib.utils import states2qubits
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, POWELL, COBYLA





num_layers = 3
num_shots = 2**11
tol = 1e-1

backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_job_size=4,
                         max_parallel_experiments=10,
                         precision="single")

filepath = "/home/archer/code/quantum/data/8x8/"
matrices = loadmat(filepath + "stiffnesses.mat")\
    ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
solutions = loadmat(filepath + "solutions.mat")['solutionData']

b = np.zeros((8,))
b[3] = -100
b[7] = 100


general_options = {}


powell_options = {'maxfev': 1e4,
           'ftol': tol,
           }

cobyla_options = {'maxiter': 0.8e4,
                  'tol':tol,
                  'disp': True}

# index = np.arange(8).reshape((-1, 1))

ansatz = FixedAnsatz(states2qubits(b.shape[0]), num_layers=num_layers)
# bounds = [(0, 2*np.pi) for _ in range(ansatz.num_parameters)]
# lb = np.zeros(ansatz.num_parameters)
# ub = np.full(fill_value=2*np.pi, shape=ansatz.num_parameters)
# bounds = Bounds(0, 2*np.pi)
#
optimizer = POWELL(**powell_options) 
optimizer = COBYLA(**cobyla_options, options=general_options)


exp = Experiment(matrices, b, optimizer, solver=VQLS, backend=backend)
exp.run()


results_path = r"/home/archer/code/quantum/experiments/vqls8x8/results/"

exp.save(results_path)


