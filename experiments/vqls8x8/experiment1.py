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
from qlib.solvers.vqls import VQLS, FixedAnsatz
from qlib.utils import states2qubits
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, POWELL

backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_parallel_experiments=20,
                         max_job_size=4,
                         precision="single")


num_layers = 3
num_shots = 2**11
tol = 1e-4

filepath = "/home/archer/code/quantum/data/8x8/"
matrices = loadmat(filepath + "stiffnesses.mat")\
    ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
solutions = loadmat(filepath + "solutions.mat")['solutionData']

b = np.zeros((8,))
b[3] = -100
b[7] = 100





options = {'maxfev': 1e4,
           'ftol': tol,
           'disp': True}


# index = np.arange(8).reshape((-1, 1))

ansatz = FixedAnsatz(states2qubits(b.shape[0]), num_layers=5)
# bounds = [(0, 2*np.pi) for _ in range(ansatz.num_parameters)]
# lb = np.zeros(ansatz.num_parameters)
# ub = np.full(fill_value=2*np.pi, shape=ansatz.num_parameters)
# bounds = Bounds(0, 2*np.pi)
#



costs = []
nfevs = []
nits = []
xs = []
for A in matrices[0:1]:
    A[ A==1 ] = 2e5
    # A = A[index, index.T]
    # b = b[index].ravel()
    # b[0] = 100
    # not1 = np.argwhere(np.diag(A)!=1)
    # B = A[not1, not1.T]
    x = np.linalg.solve(A, b)
    vqls = VQLS(A, b, ansatz=ansatz, backend=backend)

    xa = vqls.solve(optimizer=POWELL(tol=1e-4, maxfev=1e4) ).get_solution(scaled=True)
    
    ba = xa.dot(A)
    
    result = vqls.result
    costs.append(result.fun)
    nfevs.append(result.nfev)
    xs.append(result.x)
 
results_path = "/home/archer/code/quantum/experiments/results/"


np.save(results_path + "costs1", np.array(costs))
np.save(results_path + "xs1", np.array(xs))
np.save(results_path + "nfevs1", np.array(nfevs))
    

