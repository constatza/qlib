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



from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, execute

from qiskit.extensions import UnitaryGate

from scipy.linalg import expm, norm

from qlib import normalized, unitary_from_column_vector,\
    states2qubits, linear_decomposition_of_unitaries

from qiskit import Aer

backend = Aer.get_backend('statevector_simulator',
                           device='GPU',
                              max_parallel_threads=4,
                              max_parallel_experiments=4,
                               precision="single",
                              )

class BaseEvolution:
    
    def __init__(self, matrix, x0, k):
        self.matrix = matrix
        self.x0 = x0
        self.matrix_norm = norm(matrix)
        self.x0_norm = norm(x0)
        self.taylor_terms = k
        self.num_ancilla_qubits = states2qubits(k+1)
        self.num_working_qubits = states2qubits(x0.shape[0])
        self.circuit = None
        self.scale_factor = 0.0
        
        
    
    def exact_solution(self, t):
        return exact_solution(self.matrix, self.x0, t)
   
    

class UnitaryEvolution(BaseEvolution):
    
   
    def __init__(self, matrix, x0, k=3):
        super().__init__(matrix, x0, k)
        self.matrix_gate = UnitaryGate(matrix)
        self.Vs1_gate = None
        self.Ux_gate = None

        
    
    
    def build_circuit(self, include_Ux=True):

        ancilla = AncillaRegister(self.num_ancilla_qubits, name="ancilla")
        working = QuantumRegister(self.num_working_qubits, name="working")
        qc = QuantumCircuit(ancilla, working,  name="Unitary A")
        
        if include_Ux:
            qc.append(self.Ux_gate, working)
            
        qc.append(self.Vs1_gate, ancilla)
        
        for i in range(0, 2**self.num_ancilla_qubits):
            Um = self.matrix_gate.power(i).control(self.num_ancilla_qubits)
            Um.label = f"U{i}"
            Um.ctrl_state = i
            qc.append(Um, ancilla[:] + working[:])
            
        qc.append(self.Vs1_gate.inverse(), ancilla)
        
        # classical = ClassicalRegister(num_working_qubits)
        # qc.add_register(classical)
        
        self.circuit = qc
    
    
    def evolve(self, t, include_Ux=True):
        
        if include_Ux:
            self.Ux_gate = unitary_from_column_vector(self.x0, label="Ux")
        
        
        taylor_coeffs = calculate_taylor_coeffs_unitary(1, self.x0_norm,
                                                  t, 2**self.num_ancilla_qubits)
        
        self.Vs1_gate = unitary_from_column_vector(np.sqrt(taylor_coeffs), label="Vs1")
        
        self.build_circuit()
        
        self.scale_factor = taylor_coeffs.sum()
        
        return self.circuit, self.scale_factor
        
    

class RangeSimulation:

    def __init__(self, lde_evolution, backend=backend):
        self.evolution = lde_evolution
        self.scale_factors = []
        self.circuits = []
        self.solutions = None
        self.exact_solutions = []
        self.backend = backend
        

    def simulate_range(self, time_range):
        self.num_timesteps = len(time_range)
        for t in time_range:
            qc, scale_factor = self.evolution.evolve(t)
            self.circuits.append(qc)
            self.scale_factors.append(scale_factor)
        

        result = execute(self.circuits, self.backend, optimization_level=3).result()
          
        x = []
        for i in range(self.num_timesteps):
            state = result.get_statevector(i)

            x.append(state.data.real[::2**self.evolution.num_ancilla_qubits]*self.scale_factors[i])
    
        return np.array(x)


    def simulate_range_exact(self, time_range):

        for t in time_range:
            exact = self.evolution.exact_solution(t)  
            self.exact_solutions.append(exact)
            
        return np.array(self.exact_solutions)
 
    
def exact_solution(matrix, x0, t):
    return expm(matrix*t) @ x0


    
    
def calculate_taylor_coeffs_unitary(matrix_norm, x0_norm, t: float, num_ancilla_qubits: int):
    
    coeffs = []
    
    for m in range(num_ancilla_qubits):
        coeffs.append((matrix_norm*t)**m/np.math.factorial(m))
        
    return x0_norm*np.array(coeffs)



if __name__=="__main__":
  
    pass    