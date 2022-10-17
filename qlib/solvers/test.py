#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
import numpy as np
from scipy.io import loadmat
from qiskit import Aer
from qlib.utils import states2qubits
from qlib.solvers.vqls import VQLS, FixedAnsatz
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, CG, COBYLA
from qiskit.circuit.library import RealAmplitudes

backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=4,
                          max_parallel_experiments=20,
                          max_job_size=4,
                          num_shots=1,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
num_qubits = 2
size = 2**num_qubits
num_layers = 8
num_shots = 2**11
tol = 1e-6
# np.random.seed(1)

options = {'maxiter': 10000,
           'tol': tol}

for _ in range(2):
    b = np.ones(size)
    np.random.seed(1)
    A = np.random.rand(size, size)
    
    
    A = 0.5*(A + A.conj().T) 
    
    x = np.linalg.solve(A, b)
    
    
    ansatz = FixedAnsatz(states2qubits(A.shape[0]),
                       num_layers=num_layers)
    
    # ansatz = RealAmplitudes(num_qubits=num_qubits, reps=20)
    vqls = VQLS(backend=backend, 
                ansatz=ansatz)
    vqls.A = A
    vqls.b = b
    opt = COBYLA(tol=1e-4)
    
    # opt = SPSA()
    # opt = CG()
    
    xa = vqls.solve(optimizer=opt).get_solution(scaled=True)
    ba = xa.dot(A)
    print(xa)