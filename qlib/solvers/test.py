#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.extensions import UnitaryGate
from qlib.utils import states2qubits
from qlib.solvers.vqls import VQLS, FixedAnsatz
from scipy.io import loadmat
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, CG

backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=4,
                          max_parallel_experiments=4,
                          optimization_level=3,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
num_qubits = 4
size = 2**num_qubits
num_layers = 7
num_shots = 2**11
tol = 1e-8
np.random.seed(1)

options = {'maxiter': 200,
           'tol': tol,
    'disp': True}

b = np.ones(size)
np.random.seed(1)
A = np.random.rand(size, size)


A = 0.5*(A + A.conj().T) 

# filepath = "/home/archer/code/quantum/data/"
# matrices = loadmat(filepath + "stiffnessMatricesDataTraining.mat")\
#     ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
    

    
# A = matrices[np.random.randint(0, 999)]
# b = np.zeros((16,))
# b[7] = -100
# b[14] = 100 


x = np.linalg.solve(A, b)




vqls = VQLS(A, b, 
            backend=backend, 
            ansatz=FixedAnsatz(states2qubits(A.shape[0]), num_layers=num_layers))

opt = SciPyOptimizer(method='cobyla', options=options, callback=vqls.print_cost)
# opt = SPSA()
opt = CG()

xa = vqls.solve(optimizer=opt, options=options).get_solution(scaled=True)



ba = xa.dot(A)
