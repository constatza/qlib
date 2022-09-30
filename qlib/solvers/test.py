#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.extensions import UnitaryGate
from qlib.utils import unitary_from_hermitian, linear_decomposition_of_unitaries,\
    LinearDecompositionOfUnitaries
from qlib.solvers.vqls import VQLS, FixedAnsatz
from scipy.io import loadmat

backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=4,
                          optimization_level=3,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
num_qubits = 3
size = 2**num_qubits
num_layers = 3
num_shots = 2**11
tol = 1e-8

options = {'maxiter': 100,
           'tol': tol,
    'disp': True}

b = np.ones(size)
np.random.seed(1)
A = np.random.rand(size, size)
A = np.eye(size)
# filepath = "/home/archer/code/quantum/data/"
# matrices = loadmat(filepath + "stiffnessMatricesDataTraining.mat")\
#     ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
    
# A = matrices[np.random.randint(0, 999)]
A = 0.5*(A + A.conj().T) 
x = np.linalg.solve(A, b)

vqls = VQLS(A, b, 
            backend=backend, 
            ansatz=FixedAnsatz(num_qubits, num_layers=3))
xa = vqls.solve(optimizer='COBYLA', options=options).get_solution(scaled=True)

ba = xa.dot(A)