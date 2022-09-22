#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
import numpy as np
from qiskit import Aer
from vqls import VQLS, FixedAnsatz, LocalProjector, FixedAnsatz
from qlib.utils import states2qubits


backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_parallel_experiments=4,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
size = 2**4
num_layers = 3
num_shots = 2**11

b = np.ones(size)
A = np.random.rand(size, size)
x = np.linalg.solve(A, b)

vqls = VQLS(A, b)


options = {'maxiter': 50,
           'tol': 1e-4,
    'disp': True}

bounds = [(0, 2*np.pi) for _ in range(vqls.ansatz.num_parameters)]



x_approx = vqls.solve(optimizer='COBYLA',  options=options).get_solution()

b_approx = x_approx.dot(A)
