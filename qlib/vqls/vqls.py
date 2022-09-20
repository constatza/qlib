#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:12:34 2022

@author: archer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit,\
    transpile, Aer
import qiskit.circuit.library as library
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.opflow import MatrixOp, Zero, PrimitiveOp, CircuitOp, SummedOp, ComposedOp, TensoredOp
from qlib.utils import normalized, LinearDecompositionOfUnitaries, norm,\
    unitary_from_column_vector, draw
from qiskit.opflow import Z, X, I


backend = Aer.get_backend('statevector_simulator')


class VQLS:
    
    def __init__(self, A, b, ansatz, backend=backend):
        self.matrix = A
        self.target = b
        self.lcu = LinearDecompositionOfUnitaries(A)
        self.Ub = unitary_from_column_vector(b)
        self.ansatz = ansatz
        self.projector = LocalProjector(self.lcu, ansatz, Ub)

    def construct_circuits(ansatz, num_unitaries):
    
        circuits = np.empty((ansatz.num_qubits + 1, num_unitaries, num_unitaries),
                            dtype=object)
        for j in range(num_qubits+1):
            for m in range(num_unitaries):
                for n in range(num_unitaries):
                    circuits[j, m, n] = construct_term(ansatz, m, n, j=j)
                    # draw(circuits[j, m, n][1])
        return circuits
    
    
    def construct_term(self, mu, nu, j):
        opt = 3
   
        operation = self.projector.get_circuit(mu, nu, j=j)
        
        hadamard_real = HadamardTest(operation, imaginary=False).circuit
        hadamard_imag = HadamardTest(operation, imaginary=True).circuit
        
        draw(hadamard_imag)
        transpiled = transpile([hadamard_real, hadamard_imag],
                                      backend=backend,
                                      optimization_level=opt)
        # imag = qiskit.transpile(hadamard_imag, backend=backend)
    
        return transpiled
    
    
    def run_circuits(qcs, values):
        circuits = []
        for pair in qcs:
            for qc in pair:
                qc.bind_parameters(values)
                circuits.append(qc)
        binds = {p: v for (p, v) in zip(qc.parameters, values)}
        assembly = assemble(circuits, parameter_binds=[binds])
    
        return backend.run(assembly, shots=n_shots)
    
    
    def get_results(job, n_jobs):
    
        results = np.zeros(n_jobs, dtype=float)
        result = job.result()
        if backend.name() == 'statevector_simulator':
    
            for i in range(n_jobs):  # 2 parts: real, imag per term
                state = result.get_statevector(i)
                probability = state.probabilities([0])[0]
                results[i] = probability
    
        else:
            for i in range(n_jobs):  # 2 parts: real, imag per term
                counts = result.get_counts(i)
                probability = counts['0']/n_shots
                results[i] = probability
    
        return results
    
    
    def local_cost(values):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        
        
        coeffs = np.atleast_2d(c)
        L = coeffs.shape[1]
    
        # Jobs
        beta_masterjob = run_circuits(circuits[0, :, :].flatten().tolist(), values)
        delta_masterjob = run_circuits(circuits[1:, :, :].flatten().tolist(), values)
    
        # Beta results
        beta_njobs = 2*L*L
        beta_p0 = get_results(beta_masterjob, beta_njobs)
        beta_p0 = 2*beta_p0 - 1
        betas = beta_p0[:-1:2] + 1j*beta_p0[1::2]
        betas = betas.reshape((L, L))
    
        # delta results
        delta_njobs = num_qubits*beta_njobs
        delta_p0 = get_results(delta_masterjob, delta_njobs)
        delta_p0 = 2*delta_p0 - 1
        deltas = delta_p0[:-1:2] + 1j*delta_p0[1::2]
        deltas = deltas.reshape((num_qubits, L, L))
    
        delta_sum = coeffs.dot(betas + deltas.sum(axis=0)).dot(coeffs.conj().T).real[0, 0]
        norm = coeffs.dot(betas).dot(coeffs.conj().T).real[0, 0]
    
      
        return 1 - delta_sum/norm/(num_qubits-1)/2
    
    
    def optimal_state(values_opt):
        backend = qiskit.Aer.get_backend('statevector_simulator')
    
        transpiled = transpile(ansatz.reverse_bits(), optimization_level=3,
                               backend=backend)
        transpiled = transpiled.bind_parameters(values_opt)
        state = backend.run(transpiled).result().get_statevector()
    
        return np.asarray(state).real
 
class FixedAnsatz:

    def __init__(self, num_qubits, num_layers=1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.alpha = ParameterVector('a', 
                                     num_layers*4*(num_qubits-1) + num_qubits )
        
        self.qc =  QuantumCircuit(num_qubits, name='V')
        self.construct_circuit()

    
    def construct_circuit(self):
        qc = self.qc
        qubits = qc.qubits
        alpha = self.alpha
    
        ia = 0
        for _ in range(self.num_layers):
            for iz in range(0, self.num_qubits-1, 2):
                qc.cz(iz, iz+1)
                qc.ry(alpha[ia], qubits[iz])
                qc.ry(alpha[ia+1], qubits[iz+1])
                ia += 2
    
            for iz in range(1, self.num_qubits-1, 2):
                qc.cz(iz, iz+1)
                qc.ry(alpha[ia], qubits[iz])
                qc.ry(alpha[ia+1], qubits[iz+1])
                ia += 2

    @property
    def circuit(self):
        return self.qc


    
class LocalProjector:
    
    def __init__(self, lcu, ansatz, Ub):
        self.Ub = Ub
        self.lcu = lcu
        self.ansatz = ansatz
        self.control_reg = QuantumRegister(1, name='control')
        self.working_reg = QuantumRegister(lcu.gates[0].num_qubits, name='working')

        
    def get_circuit(self, mu, nu, j=0):
        control_reg = self.control_reg
        working_reg = self.working_reg
        
        inv = '\dagger'
        A_mu = self.lcu.gates[mu].control(1, label=f'$A_{mu}$')
        A_nu = self.lcu.gates[nu].inverse().control(1, 
                                                    label=f'$A^{inv:s}_{nu:d}$')
        
        num_qubits = A_mu.num_qubits
       
        qc = QuantumCircuit(control_reg, working_reg, name='$\delta_{ij}$')
        
        qc.append(self.ansatz.circuit, working_reg)
        qc.append(A_mu, 
                  [control_reg, *working_reg])
        if j > 0:
            qc.append(Ub.inverse(), working_reg)
            qc.cz(control_reg, j)
            qc.append(Ub, working_reg)
        qc.append(A_nu, [control_reg, *working_reg])
        
        return qc
    
        
    
    
def iter_callback():
    global i
    print(f"{i:d}")
    i += 1


def fun_callback(x):
    print("{:.5e}".format(local_cost(x)))


def gates2operator(operator_dict_list, num_qubits, coeffs=[1]):
    summed = []
    for i,gate_dict in enumerate(operator_dict_list):
        composed = [I^num_qubits]
        
        for (qubit, gate) in gate_dict.items():
            local_op = local_operator(CircuitOp(gate), qubit, num_qubits)
            composed.append(local_op)
            
        summed.append(ComposedOp(composed))
    
    # Except for the SummedOp, the following expression stands for Operators
    # c * Op = c**len(Op) * Operator
    # So in order to neutralize the effect:
    summed = [coeff**(1/len(Op))*Op for (Op, coeff) in zip(summed, coeffs)]
    
    return SummedOp(summed)


def local_operator(operator, qubit, num_qubits):
    tensored = []
    for i in range(num_qubits):
        if i == qubit:
            tensored.append(operator)
        else:
            tensored.append(I)
            
    return TensoredOp(tensored)
 
   
class HadamardTest:
    """Hadamard Test class for expectation values
    
    Input:
        operator: must be already controlled!!
    """
    
    def __init__(self, operator, imaginary=False, measure=False):
        self.operator = operator
        self.qc = None
        self.construct_circuit(imaginary=imaginary, measure=measure)

    def construct_circuit(self, imaginary=False, measure=False):
        
        num_working_qubits = self.operator.num_qubits - 1
       
        control_reg = QuantumRegister(1, name='control')
        working_reg = QuantumRegister(num_working_qubits, name='working')
        
        if not measure:
            qc = QuantumCircuit(control_reg, working_reg)
        else:
            
            classical_reg = ClassicalRegister(1)
            qc = QuantumCircuit(control_reg, working_reg, classical_reg)
        
        qc.h(control_reg)
        if imaginary:
            qc.sdg(control_reg)
        qc.barrier()
        qc.append(self.operator, [control_reg, *working_reg])
        qc.barrier()
        qc.h(control_reg)
        if measure:
            qc.measure(control_reg, classical_reg)
        # draw(qc.decompose())
        self.qc = qc
    
    @property
    def circuit(self):
        return self.qc


def global_hamiltonian(A, U):
    n = U.num_qubits 
    b = qi.Statevector(U.to_matrix()[:, 0])
    op = (I^n) - MatrixOp(b.to_operator()).to_pauli_op()

    return  ComposedOp([A.adjoint(), op, A]).to_pauli_op()


if __name__ == '__main__':

    b = np.array([1, 1, 1, 1])
    A = np.eye(4)
    
    Ub = unitary_from_column_vector(b)
  
    ansatz = FixedAnsatz(2, num_layers=3)  
  
    vqls = VQLS(A, b, ansatz).construct_term(1, 2, 1)