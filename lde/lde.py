#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:48:03 2022

@author: archer
"""

import sys
# sys.path is a list of absolute path strings
sys.path.append(r'/home/archer/code/quantum')


import numpy as np
import matplotlib.pyplot as plt


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate


from qlib import normalize, unitary_from_column_vector,\
    states2qubits, linear_decomposition_of_unitaries


def calculate_lde_coeffs_unitary(matrix_normalized, matrix_norm, 
                             x0_normalized, x0_norm, t: float, T: int):
    
    coeffs = []
    
    for m in range(T):
        coeffs.append((matrix_norm*t)**m/np.math.factorial(m))
        
    return x0_norm*np.array(coeffs)


def build_circuit_from_unitary(A, Vs1, Ux, t):
    
    num_working_qubits = A.num_qubits
    num_ancilla_qubits = Vs1.num_qubits
    ancilla = QuantumRegister(num_ancilla_qubits, name="ancilla")
    working = QuantumRegister(num_working_qubits, name="working")
    classical = ClassicalRegister(num_working_qubits)
    qc = QuantumCircuit(ancilla, working, classical,  name="Unitary A")
    
    
    qc.append(Vs1, ancilla)
    qc.append(Ux, working)
    
    
    for i in range(0, 2**num_ancilla_qubits):
    
        Um = A.power(i).control(num_ancilla_qubits)
        Um.label = f"U{i}"
        Um.ctrl_state = i
        qc.append(Um, ancilla[:] + working[:])
        
    qc.append(Vs1.inverse(), ancilla)
    
    
    # qc.measure(working, classical)
    
    return qc



num_working_qubits= 1

M = np.array([[0, -1],
              [1, 0]])

# M = np.eye(2**num_working_qubits)

# M = np.random.rand(2**num_working_qubits, 2**num_working_qubits)

k = 2**1 - 1

num_ancilla_qubits = states2qubits(k+1)




lcu_coeffs = np.array([.5]*4)
x0 = np.array([1, 0])
t = 20


matrix_normalized, matrix_norm = normalize(M)
x0_normalized, x0_norm = normalize(x0)

A = linear_decomposition_of_unitaries(matrix_normalized)



Agate = UnitaryGate(A[0])

Ux = unitary_from_column_vector(x0, label="Ux")



backend = Aer.get_backend('statevector_simulator',
                                  max_parallel_threads=4,
                                  max_parallel_experiments=4,
                                  precision="single")
 
x = []
for t in np.linspace(0.01, 0.6):
    lde_coeffs = calculate_lde_coeffs_unitary(matrix_normalized, matrix_norm, 
                                              x0_normalized, x0_norm,
                                              t, 2**num_ancilla_qubits)
    
    Vs1 = unitary_from_column_vector(lde_coeffs, label="Vs1")
    
    qc = build_circuit_from_unitary(Agate, Vs1, Ux, t)
    
    circuit = transpile(qc, backend, optimization_level=3)
    job = backend.run(circuit)
    state = job.result().get_statevector()
    
    result = state.probabilities_dict(np.arange(num_working_qubits))
    x.append(result['1'])


plt.plot(x)