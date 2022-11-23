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
                 max_parameters=None,
                 backend=None):

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.max_parameters = max_parameters
        self.num_parameters = None
        self.parameters = None
        self.circuit = None
        self.optimization_level = optimization_level
        self.backend = backend
        self.construct_circuit()
        self.transpile_circuit()

    def construct_ansatz(self, *args, **kwargs):
        self.construct_circuit(*args, **kwargs)
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

    def get_state(self, values_opt):
        backend = Aer.get_backend('statevector_simulator')
        qc = self.get_circuit().assign_parameters(values_opt)
        job = execute(qc, backend=backend)
        state = job.result().get_statevector()

        return np.asarray(state).real


class RealAmplitudesAnsatz(Ansatz):

    def __init__(self,
                 num_qubits,
                 num_layers=1,
                 optimization_level=3,
                 backend=None,
                 *args, **kwargs):

        super().__init__(num_qubits,
                         num_layers,
                         optimization_level,
                         backend)

        self.construct_ansatz(*args, **kwargs)

    def construct_circuit(self):
        self.circuit = RealAmplitudes(num_qubits=self.num_qubits,
                                      reps=self.num_layers)


class FixedAnsatz(Ansatz):

    def __init__(self, num_qubits,
                 num_layers=1,
                 optimization_level=3,
                 max_parameters=None,
                 backend=None):

        super().__init__(num_qubits,
                         num_layers=num_layers,
                         optimization_level=optimization_level,
                         max_parameters=max_parameters,
                         backend=backend)

        self.construct_ansatz()

    def construct_circuit(self):
        num_qubits = self.num_qubits
        num_layers = self.num_layers

        circuit = QuantumCircuit(num_qubits, name='V')
        num_parameters = 2*num_layers * \
            (num_qubits//2 + (num_qubits-1)//2) + num_qubits

        if self.max_parameters is not None:
            max_parameters = self.max_parameters
        else:
            max_parameters = num_parameters

        parameters = ParameterVector('a', max_parameters)

        num_parameters_current = 0
        for iz in range(num_qubits):
            circuit.ry(parameters[num_parameters_current], iz)
            num_parameters_current += 1

        iters = 0
        while num_parameters_current < max_parameters and iters < num_layers:
            iters += 1
            circuit, num_parameters_current = self._apply_layer(circuit,
                                                                parameters,
                                                                0, num_qubits-1,
                                                                num_parameters_current,
                                                                max_parameters)

            circuit, num_parameters_current = self._apply_layer(circuit,
                                                                parameters,
                                                                1, num_qubits-1,
                                                                num_parameters_current,
                                                                max_parameters)

        self.circuit = circuit
        return self

    def get_circuit(self):
        return self.circuit

    @staticmethod
    def _apply_layer(circuit, parameters,
                     start, end,
                     num_parameters_current,
                     max_parameters):

        qubits = circuit.qubits
        if num_parameters_current < max_parameters:
            for iz in range(start, end, 2):
                circuit.cz(iz, iz+1)
                for j in range(2):
                    circuit.ry(
                        parameters[num_parameters_current], qubits[iz+j])
                    num_parameters_current += 1

                    if num_parameters_current >= max_parameters:
                        return circuit, num_parameters_current

        return circuit, num_parameters_current


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
        A_mu = self.lcu.gates[mu].control(1, label=f'A_{mu}')
        A_nu = self.lcu.gates[nu].adjoint().control(1,
                                                    label=f'A^{inv:s}_{nu:d}')

        qc = QuantumCircuit(control_reg, working_reg,
                            name=f'δ_{mu}{nu}')
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
        # qc.save_statevector()
        self.qc = qc

    def get_circuit(self):
        return self.qc


class VQLS:

    def __init__(self,
                 A=None,
                 b=None,
                 projector=LocalProjector,
                 ansatz=None,
                 backend=backend,
                 optimizer=None,
                 optimization_level=3,
                 num_shots=1):

        self.b = b
        self.A = A
        self.ansatz = ansatz
        self.projector = projector
        if ansatz is None:
            self.num_working_qubits = None
        else:
            self.num_working_qubits = self.ansatz.num_qubits
        self.backend = backend
        self.optimization_level = optimization_level
        self.num_shots = num_shots
        self.optimizer = optimizer
        self.delete_results()
        self.delete_matrix_attrs()
        self._circuits_ready = False
        self._projector_instance = None

    def delete_matrix_attrs(self):
        self.num_unitaries = None
        self.num_jobs = None
        self.circuits = None
        self._circuits_ready = False

    def delete_results(self):
        self.result = None
        self.solution = None
        self.solution_time = None
        self.transpilation_time = None

    def construct_circuits(self):
        self.check_linear_system_exists()
        self._projector_instance = self.projector(
            self.lcu, self.ansatz, self.Ub)

        if self.ansatz is None:
            num_working_qubits = states2qubits(self.b.shape[0])
            self.num_working_qubits = num_working_qubits
            self.ansatz = FixedAnsatz(num_working_qubits,
                                      backend=backend)

        num_qubits = self.num_working_qubits + 1
        num_unitaries = self.lcu.num_unitaries
        circuits = []
        for j in range(num_qubits):
            for m in range(num_unitaries):
                for n in range(m+1):
                    real, imag = self.construct_term(m, n, j=j)
                    real, imag  = transpile([real, imag],
                            backend=self.backend,
                            )
                    circuits.append(real)
                    circuits.append(imag)

        print("# Transpiling")
        t0 = time()
        self.circuits = circuits
       # self.circuits = transpile(circuits,
       #                           backend=self.backend,
       #                           optimization_level=self.optimization_level)
        transpilation_time = time() - t0
        print_time(transpilation_time)
        self.transpilation_time = transpilation_time
        self.num_jobs = len(self.circuits)
        self._circuits_ready = False

    def construct_term(self, mu, nu, j):

        operation = self._projector_instance.get_circuit(mu, nu, j=j)

        hadamard_real = HadamardTest(operation, imaginary=False).get_circuit()
        hadamard_imag = HadamardTest(operation, imaginary=True).get_circuit()

        hadamard_real.name = f"{j}, {mu}, {nu}, real"
        hadamard_imag.name = f"{j}, {mu}, {nu}, imag"

        return (hadamard_real, hadamard_imag)

    def run_circuits(self, values):
        experiments = []

        for i in range(self.num_jobs):
            experiment = self.circuits[i].bind_parameters(values)
            job = self.backend.run(experiment)
            experiments.append(job)

        self.job = experiments
        return self

    def get_results(self, between):

        irange = range(between[0], between[1])

        results = np.zeros(len(irange), dtype=float)
        # result = self.job.result()
        job = self.job

        for i in irange:  # 2 parts: real, imag per term
            state = job[i].result().get_statevector()
            results[i-between[0]] = state.probabilities([0])[0]
        # if else:
        #     for i in irange:  # 2 parts: real, imag per term
        #         counts = job[i].result().get_counts(i)
        #         results[i-between[0]] = counts['0']/self.num_shots

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
            betas[m, m] = np.complex(1, 0)
            for l in range(m):
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

    def print_cost(self, x, *args):
        print("{:.5e}".format(self.cost))

    def solve(self, optimizer=None, initial_parameters=None, **kwargs):

        self.check_linear_system_exists()
        if not self._circuits_ready:
            self.construct_circuits()

        if optimizer is None:
            optimizer = self.optimizer

        if initial_parameters is None:
            parameters0 = np.random.rand(self.ansatz.num_parameters)
            print('Random initial parameters')
        else:
            parameters0 = initial_parameters

        print("# Optimizing")
        t0 = time()
        objective_func = self.local_cost
        
        if type(optimizer)==str:
            result = minimize(objective_func,
                              method=optimizer,
                              x0=parameters0,
                              callback=self.print_cost,
                              **kwargs)
        
        else:
            result = optimizer.minimize(objective_func,
                                        x0=parameters0,
                                        **kwargs)

        solution_time = time() - t0
        print_time(solution_time, msg="Solution")
        self.solution_time = solution_time
        self.result = result
        self.solution = self.ansatz.get_state(result.x)
        return self

    @property
    def optimal_parameters(self):
        return self.result.x

    def get_solution(self, scaled=False):
        x = self.solution
        if scaled:
            b_up_to_constant = x.dot(self.matrix)
            constants = self.target/b_up_to_constant
            constant = np.mean(constants[constants != 0])
            xopt = constant*x
        else:
            xopt = x
        return xopt

    @property
    def A(self):
        return self.matrix

    @A.setter
    def A(self, matrix):
        self.delete_matrix_attrs()
        self.delete_results()
        self.matrix = matrix
        if (matrix is not None):
            self.lcu = LinearDecompositionOfUnitaries(matrix)
        else:
            self.lcu = None

    @property
    def b(self):
        return self.target

    @b.setter
    def b(self, target):
        self.delete_results()
        self.target = target
        if target is not None:
            self.Ub = unitary_from_column_vector(target)
        else:
            self.Ub = None

    def check_linear_system_exists(self):
        if (self.lcu is None):
            raise ValueError("VQLS missing A matrix")
        elif (self.Ub is None):
            raise ValueError("VQLS missing b vector of right-hand side")


class Experiment:

    def __init__(self,
                 matrices,
                 rhs,
                 optimizer=None,
                 solver=VQLS(),
                 initial_parameter_predictor=None,
                 backend=backend):

        self.matrices = matrices
        self.target = rhs
        self.solver = solver
        self.optimizer = optimizer
        self.config = None
        self.system_size = rhs.shape[-1]



        self.data = {"Solutions": [],
                     "OptimalParameters": [],
                     "SolutionTimes": [],
                     "CostFunctionMinima": [],
                     "NumFunctionEvaluations":[],
                     "NumIterations": [],
                     "TranspilationTimes": []}

        self.initial_parameter_predictor = initial_parameter_predictor


    def run(self,
            logger=None, **kwargs):


        solver = self.solver
        solver.b = self.target
        optimizer = self.optimizer
        data = self.data

        if self.initial_parameter_predictor is None:
            num_parameters = self.solver.ansatz.num_parameters
            self.initial_parameter_predictor = SolutionPredictorLastBest(num_parameters)



        t0 = time()
        for i, A in enumerate(self.matrices):
            print("# --------------------")
            print(f'# Experiment: {i:d}')

            solver.A = A

            initial_parameters = self.initial_parameter_predictor\
                .predict_solution(y=data['OptimalParameters'])

            if i>0 and not (type(optimizer)==str  or optimizer is None):
                optimizer.set_options(**kwargs)

            solver.solve(optimizer=optimizer,
                         initial_parameters=initial_parameters, 
                         **kwargs)
            

            result = solver.result
            data['NumIterations'].append(result.nit)
            data['CostFunctionMinima'].append(result.fun)
            data['NumFunctionEvaluations'].append(result.nfev)
            data['OptimalParameters'].append(result.x)
            data['Solutions'].append(solver.get_solution(scaled=True))
            data['TranspilationTimes'].append(solver.transpilation_time)
            data['SolutionTimes'].append(solver.solution_time)

            if logger is not None:
                # Must be the same order as filenames above!!
                logger.save([array[-1] for array in data.values()])

            print(f"# Function Value: {solver.result.fun:1.5e}")
            print_time(time() - t0, msg="Total Simulation")


        self.data = {key: np.array(value) for key, value in data.items()}

        return self


class SolutionPredictor:

    def __init__(self, size):
        self.size = (size,)
        self.iteration = -1

    def predict_solution(self, *args, **kwargs):
        self.iteration += 1
        return self.predict( *args, **kwargs)

    def predict(self):
        pass


class SolutionPredictorRandom(SolutionPredictor):

    def __init__(self, size):
        super().__init__(size)

    def predict(self, *args, **kwargs):
        return np.random.rand(self.size)


class SolutionPredictorConstant(SolutionPredictor):

    def __init__(self, size, value=0):
        super().__init__(size)
        self.value = value

    def predict(self, *args, **kwargs):
        return np.full(self.size, self.value)


class SolutionPredictorLastBest(SolutionPredictor):

    def __init__(self, size):
        super().__init__(size)


    def predict(self, y, *args, **kwargs):

        if self.iteration>0:
            return y[self.iteration-1]
        else:
            return np.random.rand(*self.size)


class SolutionPredictorSurrogate(SolutionPredictor):

    def __init__(self, model, X):
        size = model.layers[-1].output.shape[-1]
        super().__init__(size)
        self.model = model
        self.X = X

    def predict(self, *args, **kwargs):
        return self.model.predict(self.X[self.iteration])







if __name__ == '__main__':
    pass
