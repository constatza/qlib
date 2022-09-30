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
from scipy.io import loadmat

backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_parallel_experiments=4,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
num_qubits = 4
size = 2**num_qubits
num_layers = 3
num_shots = 2**11
tol = 1e-8

b = np.ones(size)
A = np.random.rand(size, size)
# A[ np.abs(A) < tol ] = 0








filepath = "/home/archer/code/quantum/data/"
matrices = loadmat(filepath + "stiffnessMatricesDataTraining.mat")\
    ['stiffnessMatricesData'].transpose(2, 0, 1).astype(np.float64)
    
A = matrices[np.random.randint(0, 999)]
A = 0.5*(A + A.conj().T) 
x = np.linalg.solve(A, b)




lcu = LinearDecompositionOfUnitaries(A)
print(lcu.valid_decomposition())
F1 = lcu.decomposition[0]
g = F1 @ F1.conj().T

U = UnitaryGate(F1, label="why qiskit?")

qc = QuantumCircuit(num_qubits + 1)

qc.append(U, range(num_qubits))

cu = U.control(1)

qc.append(cu, range(num_qubits + 1))

print(qc)
