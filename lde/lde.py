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

from scipy.linalg import expm

from qlib import normalize, unitary_from_column_vector,\
    states2qubits, linear_decomposition_of_unitaries
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace


def calculate_lde_coeffs_unitary(matrix_norm, x0_norm, t: float, num_ancilla_qubits: int):
    
    coeffs = []
    
    for m in range(num_ancilla_qubits):
        coeffs.append((matrix_norm*t)**m/np.math.factorial(m))
        
    return x0_norm*np.array(coeffs)


def build_circuit_from_unitary(A, Vs1, Ux):
    
    num_working_qubits = A.num_qubits
    num_ancilla_qubits = Vs1.num_qubits
    ancilla = QuantumRegister(num_ancilla_qubits, name="ancilla")
    working = QuantumRegister(num_working_qubits, name="working")
    qc = QuantumCircuit(ancilla, working,  name="Unitary A")
    
    qc.append(Ux, working)
    qc.append(Vs1, ancilla)
  
    for i in range(0, 2**num_ancilla_qubits):
        Um = A.power(i).control(num_ancilla_qubits)
        Um.label = f"U{i}"
        Um.ctrl_state = i
        qc.append(Um, ancilla[:] + working[:])
        
    qc.append(Vs1.inverse(), ancilla)
    
    # classical = ClassicalRegister(num_working_qubits)
    # qc.add_register(classical)
    
    return qc


def evolve_unitary(unitary_matrix, x0, t, num_ancilla_qubits):
    
    x0_normalized, x0_norm = normalize(x0)
    matrix_normalized, matrix_norm = normalize(unitary_matrix)
    Ux = unitary_from_column_vector(x0, label="Ux")
    
    lde_coeffs = calculate_lde_coeffs_unitary(1, x0_norm,
                                              t, 2**num_ancilla_qubits)
    
    Vs1 = unitary_from_column_vector(np.sqrt(lde_coeffs), label="Vs1")
    
    qc = build_circuit_from_unitary(unitary_matrix, Vs1, Ux)
    
    C = lde_coeffs.sum()
    
    return qc, C


def exact_solution(matrix, x0, t):
    
    return expm(matrix*t) @ x0
    
    


if __name__=="__main__":
    num_working_qubits= 1
    
    M = np.array([[0, -1],
                  [1, 0]])
    
    # M = np.eye(2**num_working_qubits)
    
    # M = np.random.rand(2**num_working_qubits, 2**num_working_qubits)
    
    k = 2**3 - 1
    
    num_ancilla_qubits = states2qubits(k+1)
    total_qubits = num_ancilla_qubits + num_working_qubits
  
   
    lcu_coeffs = np.array([.5]*4)
    x0 = np.array([1, 0])
   
  
    matrix_normalized, matrix_norm = normalize(M)
    
    A = linear_decomposition_of_unitaries(matrix_normalized)
    
    
    Agate = UnitaryGate(M)
    
    
    backend = Aer.get_backend('statevector_simulator',
                                   device='GPU',
                                      max_parallel_threads=4,
                                      max_parallel_experiments=4,
                                       precision="single",
                                      )
     
    
    time = np.linspace(0.01, 3)
    
    experiments = []
    classicals = []
    scale_factor = []
    for t in time:
        
        qc, C = evolve_unitary(Agate, x0, t, num_ancilla_qubits)
        
        x0, _ = normalize(x0)
        exact = exact_solution(M, -x0, t)  
        
        classicals.append(exact)
        experiments.append(qc)
        scale_factor.append(C)
        
        DM=DensityMatrix.from_instruction(qc)

        # PT=partial_trace(DM, np.arange(0,num_ancilla_qubits+1))

        state = Statevector.from_instruction(qc)
    
    
    result = execute(experiments, backend, optimization_level=3).result()
   
    x = []
    for i in range(len(time)):
        state = result.get_statevector(i)
        # x.append(result['1'])
        x.append(state.data.real[::2**num_ancilla_qubits]*scale_factor[i])
        
        

        
    x = np.array(x)
    x_classical = np.array(classicals)
    
    
    lines = plt.plot(time, x, time, -x_classical)
    plt.title(f"Taylor terms k={2**num_ancilla_qubits:d}")
    plt.xlabel('t')
    
    plt.legend(iter(lines), ('$\dot x$', '$x$', 'exact $\dot x$', 'exact $x$'))





