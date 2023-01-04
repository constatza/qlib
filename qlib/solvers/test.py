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
from qlib.solvers.vqls import VQLS, FixedAnsatz, RealAmplitudesAnsatz 
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, CG, COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer.backends.aer_simulator import AerSimulator

backend = Aer.get_backend('statevector_simulator',
                         precision="single")


num_qubits = 1
num_layers = 2

size = 2**num_qubits
tol = 1e-8
np.random.seed(1)

N = 2**num_qubits
matrices = np.random.rand(2, N, N)
matrices = 0.5*(matrices + matrices.transpose(0, 2, 1))
b = np.random.rand(N,1)


ansatz = FixedAnsatz(num_qubits,
                   num_layers=num_layers,
                   max_parameters=17)
qc = ansatz.get_circuit()
print(qc)

vqls = VQLS(backend=backend,
            ansatz=ansatz)

options = {'maxiter': 5000,
           'tol': tol,
           'callback':vqls.print_cost,
           'rhobeg':1e-1}


from qiskit.opflow import I, X, H, Z

for A in matrices[0:1]:

    op = H
    x = np.linalg.solve(op.to_matrix(), b)

    vqls.A = H  
    vqls.b = b

    xa = vqls.solve(optimizer='BFGS',
                    ).get_solution(scaled=True)
    print(xa - x.ravel())
