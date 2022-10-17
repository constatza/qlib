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
        num_qubits = self.num_qubits
        num_layers = self.num_layers

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

        qc = QuantumCircuit(control_reg, working_reg,
                            name='$\delta_{\mu \nu}$')
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

    def __init__(self,
                 A=None,
                 b=None,
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
        self.transpilation_time = None

        self.construct_circuits()

    def construct_circuits(self):
        assert (self.A is not None) and (self.b is not None)
        num_qubits = self.num_working_qubits + 1
        num_unitaries = self.lcu.num_unitaries
        circuits = []
        for j in range(num_qubits):
            for m in range(num_unitaries):
                for n in range(m+1):
                    real, imag = self.construct_term(m, n, j=j)
                    circuits.append(real)
                    circuits.append(imag)

        
        print("# Transpiling")
        t0 = time()
        self.circuits = transpile(circuits,
                                  backend=self.backend,
                                  optimization_level=self.optimization_level)

        transpile_time = time() - t0
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
        return self.cost

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
        t0 = time()
        objective_func = self.local_cost
        if type(optimizer) is str:
            result = minimize(objective_func,
                              method=optimizer,
                              x0=parameters0,
                              callback=self.print_cost,
                              **kwargs)
        else:
            result = optimizer.minimize(objective_func,
                                        parameters0)

        solution_time = time() - t0
        print_time(solution_time, msg="Solution")
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
        return xopt


class Experiment:

    def __init__(self,
                 matrices,
                 rhs,
                 optimizer,
                 solver=VQLS,
                 backend=backend):

        self.matrices = matrices
        self.target = rhs
        self.solver = solver
        self.optimizer = optimizer
        self.backend = backend
        self.func_costs = None
        self.num_iterations = None
        self.num_func_evals = None
        self.solutions = None
        self.solution_times = None
        self.transpilation_times = None

    def run(self):

        b = self.target
        backend = self.backend
        optimizer = self.optimizer

        num_iterations = []
        num_func_evals = []
        func_costs = []
        solutions = []
        transpilation_times = []
        solution_times = []
        t0 = time()
        for i,A in enumerate(self.matrices[0:1]):
            print("# --------------------")
            print(f'# Experiment: {i:d}')
            
            solver = self.solver
            solver.A = A

            solver.solve().get_solution(scaled=True)

            num_iterations.append(solver.result.nit)
            func_costs.append(solver.result.fun)
            num_func_evals.append(solver.result.nfev)

            solutions.append(solver.get_solution(scaled=True))
            transpilation_times.append(solver.transpilation_time)
            solution_times.append(solver.solution_time)
            

            print_time(time() - t0, msg="Total Simulation")

        self.func_costs = np.array(num_iterations)
        self.num_iterations = np.array(num_iterations)
        self.num_func_evals = np.array(num_func_evals)
        self.solutions = np.array(solutions)
        self.solution_times = np.array(solution_times)
        self.transpilation_times = np.array(transpilation_times)

        return self

    def save(self, results_path):
        """Save experiments as .npy binaries"""
        from datetime import datetime
        
        suffix = datetime.today().strftime("_%Y-%m-%d_%H-%M")
        
        names = {"Solutions": self.solutions,
                "SolutionTimes": self.solution_times,
                "MinFunctionValues": self.func_costs,
                "NumFunctionEvaluations": self.num_func_evals,
                "NumIterations": self.num_iterations}

        for name, array in names.items():
            np.save(results_path + name + suffix, array)


if __name__ == '__main__':

   pass
