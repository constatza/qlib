#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:12:34 2022

@author: archer
"""

import numpy as np
from time import time
from scipy.optimize import minimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, \
    transpile, Aer, execute
from qiskit.circuit import ParameterVector
from qlib.utils import LinearDecompositionOfUnitaries, print_time, \
    unitary_from_column_vector, states2qubits
from qiskit.circuit.library import RealAmplitudes

backend = Aer.get_backend('statevector_simulator')



class Ansatz:

    def __init__(self, 
                 num_qubits, 
                 num_layers=1, 
                 optimization_level=3, 
                 backend=None):
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_parameters = None
        self.parameters = None
        self.circuit = None
        self.optimization_level = optimization_level
        self.backend = backend
        self.construct_circuit()
        self.transpile_circuit()
    
    def construct_ansatz(self):
        self.construct_circuit()
        self.num_parameters = self.circuit.num_parameters
        self.parameters = self.circuit.parameters
        self.transpile_circuit()
    
    def construct_circuit(self):
        pass
    
    def transpile_circuit(self):
        self.circuit = transpile(self.circuit, 
                            optimization_level=self.optimization_level, 
                            backend=self.backend)

    def get_circuit(self):
        return self.circuit



class RealAmplitudesAnsatz(Ansatz):
    
    def __init__(self, 
                 num_qubits, 
                 num_layers=1, 
                 optimization_level=3, 
                 backend=None):
        
        super().__init__(num_qubits,
                         num_layers,
                         optimization_level,
                         backend)
        
        self.construct_ansatz()
    
   
        
    def contruct_circuit(self):
        self.circuit = RealAmplitudes(num_qubits=self.num_qubits, 
                                 reps=self.num_layers)
        
    
        

class FixedAnsatz(Ansatz):

    def __init__(self, num_qubits, 
                 num_layers=1, 
                 optimization_level=3, 
                 backend=None):
        
        super().__init__(num_qubits, 
                       num_layers, 
                       optimization_level, 
                       backend)
        
    
        self.construct_ansatz()

    def construct_circuit(self):
        num_qubits =self.num_qubits
        num_layers =self.num_layers
        
        circuit = QuantumCircuit(num_qubits, name='V')
        num_parameters = 2*num_layers * \
            (num_qubits//2 + (num_qubits-1)//2) + num_qubits
        qubits = circuit.qubits
        parameters = ParameterVector('a', num_parameters)

        ia = 0
        for iz in range(num_qubits):
            circuit.ry(parameters[ia], iz)
            ia += 1

        for _ in range(num_layers):
            for iz in range(0, num_qubits-1, 2):
                circuit.cz(iz, iz+1)
                circuit.ry(parameters[ia], qubits[iz])
                circuit.ry(parameters[ia+1], qubits[iz+1])
                ia += 2

            for iz in range(1, num_qubits-1, 2):
                circuit.cz(iz, iz+1)
                circuit.ry(parameters[ia], qubits[iz])
                circuit.ry(parameters[ia+1], qubits[iz+1])
                ia += 2
                
        self.circuit = circuit

    def get_circuit(self):
        return self.circuit



class LocalProjector:

    def __init__(self, lcu, ansatz, Ub):
        self.Ub = Ub
        self.lcu = lcu
        self.ansatz = ansatz
        self.control_reg = QuantumRegister(1, name='control')
        self.working_reg = QuantumRegister(
            lcu.gates[0].num_qubits, name='working')
        self.num_working_qubits = self.working_reg.size

    def get_circuit(self, mu, nu, j=0):
        control_reg = self.control_reg
        working_reg = self.working_reg

        inv = '\dagger'
        A_mu = self.lcu.gates[mu].control(1, label=f'$A_{mu}$')
        A_nu = self.lcu.gates[nu].adjoint().control(1,
                                                    label=f'$A^{inv:s}_{nu:d}$')

        qc = QuantumCircuit(control_reg, working_reg, name='$\delta_{\mu \nu}$')
        try:
            ansatz_circuit = self.ansatz.get_circuit()
        except AttributeError:
            ansatz_circuit = self.ansatz
            
        qc.append(ansatz_circuit, working_reg)
        qc.append(A_mu, [control_reg, *working_reg])
        if j > 0:
            qc.append(self.Ub.adjoint(), working_reg)
            qc.cz(control_reg, j)
            qc.append(self.Ub, working_reg)
        qc.append(A_nu, [control_reg, *working_reg])

        return qc


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

    def get_circuit(self):
        return self.qc
    

class VQLS:

    def __init__(self, A, b, 
                 initial_parameters=None,
                 projector=LocalProjector, 
                 ansatz=None, 
                 backend=backend, 
                 optimizer=None, 
                 optimization_level=3, 
                 num_shots=1):
        self.matrix = A
        self.target = b
        self.lcu = LinearDecompositionOfUnitaries(A)
        self.Ub = unitary_from_column_vector(b)
        if ansatz is None:
            self.ansatz = FixedAnsatz(states2qubits(A.shape[0]), 
                                                    backend=backend)
        else:
            self.ansatz = ansatz
            
        self.projector = projector(self.lcu, self.ansatz, self.Ub)
        self.num_working_qubits = self.projector.num_working_qubits
        self.backend = backend
        self.optimization_level = optimization_level
        self.num_shots = num_shots
        self.optimizer = optimizer
        self.initial_parameters = initial_parameters
        
        self.circuits = None
        self.num_unitaries = None
        self.num_jobs = None 
        self.result = None
        self.solution = None
        self.solution_time = None
        self.transpile_time = None

        self.construct_circuits()
        
    def construct_circuits(self):
        num_qubits = self.num_working_qubits + 1
        num_unitaries = self.lcu.num_unitaries
        circuits = []
        for j in range(num_qubits):
            for m in range(num_unitaries):
                for n in range(m+1):
                    real, imag = self.construct_term(m, n, j=j)
                    circuits.append(real)
                    circuits.append(imag)
        
        t1 = time()
        print("# Transpiling")
        self.circuits = transpile(circuits, 
                                  backend=self.backend, 
                                  optimization_level=self.optimization_level)
        print("# End")
        t2 = time()
        transpile_time = t2 - t1
        print_time(transpile_time)
        self.transpile_time = transpile_time
        self.num_jobs = len(self.circuits)
  

    def construct_term(self, mu, nu, j):

        operation = self.projector.get_circuit(mu, nu, j=j)

        hadamard_real = HadamardTest(operation, imaginary=False).get_circuit()
        hadamard_imag = HadamardTest(operation, imaginary=True).get_circuit()
        
        hadamard_real.name = f"{j}, {mu}, {nu}, real"
        hadamard_imag.name = f"{j}, {mu}, {nu}, imag"

        return (hadamard_real, hadamard_imag)

    def run_circuits(self, values):
        experiments = []
  
        for i in range(self.num_jobs):
            experiment = self.circuits[i].bind_parameters(values)
            experiments.append(experiment)
                
        self.job = self.backend.run(experiments)

        return self

    def get_results(self, between):
        irange = range(between[0], between[1])

        results = np.zeros(len(irange), dtype=float)
        result = self.job.result()

        if self.backend.name() == 'statevector_simulator':
            for i in irange:  # 2 parts: real, imag per term
                state = result.get_statevector(i)
                results[i-between[0]] = state.probabilities([0])[0]
        else:
            for i in irange:  # 2 parts: real, imag per term
                counts = result.get_counts(i)
                results[i-between[0]] = counts['0']/self.num_shots

        return results

    def local_cost(self, values):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        num_working_qubits = self.num_working_qubits
        coeffs = np.atleast_2d(self.lcu.coeffs)
        L = coeffs.shape[1]

        # Jobs
        self.run_circuits(values)
        num_jobs_beta = (L**2 - L) + 2*L

        # Beta results
        index_jobs_beta = (0, num_jobs_beta)
        beta_p0 = self.get_results(index_jobs_beta)
        beta_p0 = 2*beta_p0 - 1
        betas_unique = beta_p0[:-1:2] + 1j*beta_p0[1::2]
        betas = np.zeros((L, L), dtype=np.complex128)
        for m in range(L):
            for l in range(m+1):
                betas[m, l] = betas_unique[m + l]
                if l < m:
                    betas[l, m] = betas_unique[m + l].conj()

        # delta results
        num_jobs_delta = num_working_qubits*num_jobs_beta
        index_jobs_delta = (num_jobs_beta, num_jobs_beta + num_jobs_delta)
        delta_p0 = self.get_results(index_jobs_delta)
        delta_p0 = 2*delta_p0 - 1
        deltas_unique = delta_p0[:-1:2] + 1j*delta_p0[1::2]
        deltas = np.zeros((num_working_qubits, L, L), dtype=np.complex128)
        
        experiment_id = 0
        for j in range(num_working_qubits):
            for m in range(L):
                for l in range(m+1):
                    deltas[j, m, l] = deltas_unique[experiment_id]
                    if l < m:
                        deltas[j, l, m] = deltas_unique[experiment_id].conj()
                    experiment_id += 1
                    
        delta_sum = coeffs.dot(deltas.sum(axis=0)
                               ).dot(coeffs.conj().T).real[0, 0]
        beta_norm = coeffs.dot(betas).dot(coeffs.conj().T).real[0, 0]

        self.cost = 1/2 - delta_sum/beta_norm/num_working_qubits/2
        return np.sqrt(self.cost)

    def optimal_state(self, values_opt):
        backend = Aer.get_backend('statevector_simulator')
        qc = self.ansatz.get_circuit().assign_parameters(values_opt)
        job = execute(qc, backend=backend)

        state = job.result().get_statevector()

        return np.asarray(state).real

    def print_cost(self, x):
        print("{:.5e}".format(self.cost))  
    
    def solve(self, optimizer=None, **kwargs):
        if optimizer is None:
            try:
                optimizer = self.optimizer
            except:
                raise ValueError("No optimizer")
        
        if self.initial_parameters is None:
            parameters0 = np.random.rand(self.ansatz.num_parameters)
        else:
            parameters0 = self.initial_parameters
            
        print("# Optimizing")
        t1 = time()
        if type(optimizer) is str:
            result = minimize(self.local_cost,
                        method=optimizer,
                        x0=parameters0,
                        callback=self.print_cost,
                        **kwargs)
        else:
            result = optimizer.minimize(self.local_cost, 
                                        parameters0)
        print("# End")
        t2 = time()
        solution_time = t2 - t1
        print_time(solution_time)
        self.solution_time = solution_time
        self.result = result
        self.solution = self.optimal_state(result.x)
        return self
    
    def get_solution(self, scaled=False):
        x = self.solution
        if scaled:
            b_up_to_constant = x.dot(self.matrix)
            constant = np.mean(self.target/b_up_to_constant)
            xopt = constant*x
        else:
            xopt = x
            
        print("# approx", xopt)
        
        return xopt



if __name__ == '__main__':

   pass
