#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:38:15 2022

@author: archer
"""

import numpy as np
from qiskit import Aer
from qlib.solvers.vqls import VQLS
from scipy.io import loadmat



backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_parallel_experiments=4,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
size = 2**2
num_layers = 3
num_shots = 2**11
tol = 1e-9

filepath = "/home/archer/code/quantum/data/"
matrices = loadmat(filepath + "stiffnessMatricesDataTraining.mat")\
    ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
solutions = loadmat(filepath + "solutionDataTraining.mat")['solutionData']

b = np.zeros((16,))
b[7] = -100
b[14] = 100 



options = {'maxiter': 50,
           'tol': 1e-10,
    'disp': True}

# bounds = [(0, 2*np.pi) for _ in range(vqls.ansatz.num_parameters)]

for A in matrices[0:2]:
    x = np.linalg.solve(A, b)
    vqls = VQLS(A, b)

    xa = vqls.solve(optimizer='COBYLA',  
                          options=options).get_solution(scaled=True)
    
    ba = xa.dot(A)
    