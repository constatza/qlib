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


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, execute
from qiskit import Aer
from qiskit.compiler import transpile, assemble
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate


from qlib import normalize, unitary_from_column_vector,\
    states2qubits, linear_decomposition_of_unitaries


def calculate_lde_coeffs_unitary(matrix_norm, x0_norm, t: float, num_ancilla_qubits: int):
    
    coeffs = []
    
    for m in range(num_ancilla_qubits):
        coeffs.append((matrix_norm*t)**m/np.math.factorial(m))
        
    return x0_norm*np.array(coeffs)


def build_circuit_from_unitary(A, Vs1, Ux):
    
    num_working_qubits = A.num_qubits
    num_ancilla_qubits = Vs1.num_qubits
    ancilla = AncillaRegister(num_ancilla_qubits, name="ancilla")
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
    
    
    return qc


def evolve_unitary(unitary_matrix, x0, t, num_ancilla_qubits):
    
    x0_normalized, x0_norm = normalize(x0)
    Ux = unitary_from_column_vector(x0, label="Ux")
    
    lde_coeffs = calculate_lde_coeffs_unitary(1, x0_norm,
                                              t, 2**num_ancilla_qubits)
    
    Vs1 = unitary_from_column_vector(lde_coeffs, label="Vs1")
    
    qc = build_circuit_from_unitary(unitary_matrix, Vs1, Ux)
    
    return qc


if __name__=="__main__":
    num_working_qubits= 1
    
    M = np.array([[0, -1],
                  [1, 0]])
    
    # M = np.eye(2**num_working_qubits)
    
    # M = np.random.rand(2**num_working_qubits, 2**num_working_qubits)
    
    k = 2**2 - 1
    
    num_ancilla_qubits = states2qubits(k+1)
    total_qubits = num_ancilla_qubits + num_working_qubits
  
   
    lcu_coeffs = np.array([.5]*4)
    x0 = np.array([1, 1])
   
  
    matrix_normalized, matrix_norm = normalize(M)
    
    A = linear_decomposition_of_unitaries(matrix_normalized)
    
    
    Agate = UnitaryGate(M)
    
    
    backend = Aer.get_backend('statevector_simulator',
                                   device='GPU',
                                      max_parallel_threads=4,
                                      max_parallel_experiments=4,
                                       precision="single",
                                      )
     
    
    time = np.linspace(0.01, 4)
    
    experiments = []
    for t in time:
        
        qc = evolve_unitary(Agate, x0, t, num_ancilla_qubits)
        
        experiments.append(qc)
    
    
    result = execute(experiments, backend, optimization_level=3).result()
   
    x = []
    for i in range(len(time)):
        state = result.get_statevector(i)
        
        
        # x.append(result['1'])
        x.append(np.asarray(state).real[::2**num_ancilla_qubits])
        
    x = np.array(x)
    
    lines = plt.plot(time, x)
    plt.title(f"Taylor terms k={2**num_ancilla_qubits:d}")
    plt.xlabel('t')
    
    plt.legend(iter(lines), ('$\dot x$', '$x$'))




