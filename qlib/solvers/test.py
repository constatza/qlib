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
num_qubits = 2
size = 2**num_qubits
num_layers = 3
num_shots = 2**11
tol = 1e-8
np.random.seed(1)

b = np.ones(size)
A = np.random.rand(size, size)
A = np.eye(size)
# A[ np.abs(A) < tol ] = 0


options = {'maxiter': 50,
           'tol': 1e-10,
    'disp': True}

# bounds = [(0, 2*np.pi) for _ in range(vqls.ansatz.num_parameters)]


A = 0.5*(A + A.conj().T) 
x = np.linalg.solve(A, b)


x = np.linalg.solve(A, b)
vqls = VQLS(A, b)

xa = vqls.solve(optimizer='COBYLA',  
                      options=options).get_solution(scaled=True)

ba = xa.dot(A)