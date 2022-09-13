#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:48:03 2022

@author: archer
"""

import sys
# sys.path is a list of absolute path strings
sys.path.append(r'/home/archer/code/quantum')

from qiskit import Aer
from qlib import normalized, unitary_from_column_vector,\
    states2qubits, linear_decomposition_of_unitaries
from scipy.linalg import expm, norm
from qiskit.extensions import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, execute
import numpy as np




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
        self.matrix_norm = norm(matrix, ord=2)
        self.x0_norm = norm(x0)
        self.num_working_qubits = states2qubits(x0.shape[0])
        self.num_taylor_coeffs = 2**states2qubits(k+1)
        self.taylor_coeffs = None
        self.circuit = None
        self.Vs1_gate = None
        self.Ux_gate = None

    def exact_solution(self, t):
        return exact_solution(self.matrix, self.x0, t)


    def evolve(self, t, include_Ux=True):
    
        if include_Ux:
            self.Ux_gate = unitary_from_column_vector(self.x0, label="Ux")
    
        self.calculate_taylor_coeffs(t)
    
        self.construct_gates()
    
        self.build_circuit()
    
        return self.circuit, self.scale_factor
    
    def calculate_taylor_coeffs(self, t: float):
    
        coeffs = []
        
        for m in range(self.num_taylor_coeffs):
            coeffs.append((self.matrix_norm*t)**m/np.math.factorial(m))
    
        self.taylor_coeffs = self.x0_norm*np.array(coeffs)

    def construct_gates(self):

        raise NotImplementedError("Method must be overloaded")
        
        
    @property
    def scale_factor(self):
        
        raise NotImplementedError("Method must be overloaded")
        


class UnitaryEvolution(BaseEvolution):

    def __init__(self, matrix, x0, k=3):
        super().__init__(matrix, x0, k)
        self.matrix_gate = UnitaryGate(matrix)
        self.num_ancilla_qubits =   int(states2qubits(k+1))
    
        

    def build_circuit(self, include_Ux=True):

        ancilla = AncillaRegister(self.num_ancilla_qubits, name="ancilla")
        working = QuantumRegister(self.num_working_qubits, name="working")
        qc = QuantumCircuit(working, ancilla, name="Unitary A")

        if include_Ux:
            qc.append(self.Ux_gate, working)
  
        qc.append(self.Vs1_gate, ancilla)

        for i in range(1, self.num_taylor_coeffs):
            Um = self.matrix_gate.copy().power(i).control(self.num_ancilla_qubits)
            Um.label = f"U{i}"
            Um.ctrl_state = i
            qc.append(Um, [*ancilla, *working])

        qc.append(self.Vs1_gate.inverse(), ancilla)

        # classical = ClassicalRegister(num_working_qubits)
        # qc.add_register(classical)

        self.circuit = qc

    
    def construct_gates(self):
        
        self.Vs1_gate = unitary_from_column_vector(np.sqrt(self.taylor_coeffs),
                                   label="Vs1")
    @property
    def scale_factor(self):
        
        return self.taylor_coeffs.sum()

class Evolution(BaseEvolution):

    def __init__(self, matrix, x0, k=3):
        super().__init__(matrix, x0, k)
        # convert matrix to linear combination of unitaries
        # coefficients are 0.5 for this algorithm
        self.matrix_normalized = self.matrix/self.matrix_norm
        matrices, coeffs = linear_decomposition_of_unitaries(self.matrix_normalized)
        self.lcu_matrices = matrices
        self.lcu_coeffs = coeffs
        self.lcu_gates = list(map(UnitaryGate, matrices))
        self.num_of_unitaries = len(self.lcu_gates)
        self.num_ancilla_qubits =  k
        self.num_decomposition_qubits = states2qubits(self.num_of_unitaries)



    def build_circuit(self, include_Ux=True):

        ancilla_main = AncillaRegister(self.num_ancilla_qubits, name="ancilla")
        working = QuantumRegister(self.num_working_qubits, name="working")
        
        ancilla_decomposition = []
        
        for i in range(self.num_ancilla_qubits):
            subreg =  AncillaRegister(self.num_decomposition_qubits, 
                                   name=f"decomposition_{i:d}") 
            
            ancilla_decomposition.append(subreg)
            
        
        qc = QuantumCircuit(working,  ancilla_main, *ancilla_decomposition,
                            name="LDE")

        if include_Ux:
            qc.append(self.Ux_gate, working)

        qc.append(self.Vs1_gate, ancilla_main)
        
        for m, anc in enumerate(ancilla_main):
            qc.append(self.Va_gate, ancilla_decomposition[m])
            for i in range(0, self.num_of_unitaries ):
                Ai = self.lcu_gates[i].control(self.num_decomposition_qubits + 1)
                Ai.label = f"A{i}"
                Ai.ctrl_state = 2*i + 1
                
                qc.append(Ai, [anc, *ancilla_decomposition[m], working] )
                
        for reg in ancilla_decomposition:
            qc.append(self.Va_gate.inverse(), reg)
            
        self.circuit = qc

            
    def construct_Vs1_gate(self):
        k = self.num_ancilla_qubits
        total_size = 2**k
        array = np.zeros(total_size)
        for j in range(k+1):
            array[total_size - 2**(k-j)] = np.sqrt(self.taylor_coeffs[j])
            
        self.Vs1_gate = unitary_from_column_vector(array,
                                   label="Vs1")
        
    def construct_Va_gate(self):
        
        self.Va_gate = unitary_from_column_vector(np.sqrt(np.array(self.lcu_coeffs)),
                                                  label="Va")
    
    def construct_gates(self):
        self.construct_Vs1_gate()
        self.construct_Va_gate()

    
    @property
    def scale_factor(self):
        alpha = np.sum(np.array(self.lcu_coeffs))
        return np.dot(alpha**np.arange(self.num_taylor_coeffs),
                      self.taylor_coeffs)


    
class RangeSimulation:

    def __init__(self, lde_evolution, backend=backend):
        self.evolution = lde_evolution
        self.scale_factors = []
        self.circuits = []
        self.backend = backend

    def simulate_range(self, time_range, apply_scale=True):
        self.circuits = []

        num_timesteps = len(time_range)

        for t in time_range:
            qc, scale_factor = self.evolution.evolve(t)
            self.circuits.append(qc)
            self.scale_factors.append(scale_factor)

        result = execute(self.circuits, self.backend,
                         optimization_level=3).result()
        
        x = []
        
        for i in range(num_timesteps):
            state = result.get_statevector(i)
            scale = self.scale_factors[i]
            solution = state.data.real[:2**self.evolution.num_working_qubits]
            if not apply_scale:
                scale = 1
            
            x.append(solution *scale)

        return np.array(x)

    def simulate_range_exact(self, time_range):
        exact_solutions = []
        for t in time_range:
            exact = self.evolution.exact_solution(t)
            exact_solutions.append(exact)

        return np.array(exact_solutions)


def exact_solution(matrix, x0, t):
    return expm(matrix*t) @ x0





if __name__ == "__main__":

    pass
