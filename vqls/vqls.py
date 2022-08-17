#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:12:34 2022

@author: archer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import qiskit
import qiskit.circuit.library as library
import qiskit.quantum_info as qi
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.opflow import I, H, Z, X, MatrixOp, Zero, PrimitiveOp, CircuitOp, SummedOp, ComposedOp, TensoredOp


def hadamard_test(operator, imaginary=False):
    num_qubits = operator.num_qubits
    qc = QuantumCircuit(num_qubits, 1)
    qubits = qc.qubits
    qc.h(0)
    if imaginary:
        qc.sdg(0)
    qc.barrier()
    qc.append(operator, qubits)
    qc.barrier()
    qc.h(0)
    if backend.name() == 'qasm_simulator':
        qc.measure(0, 0)
    # draw(qc.decompose())
    return qc


def local_projector(num_qubits, mu, nu, j=0):
    qc = QuantumCircuit(num_qubits+1, name='P')
    qubits = qc.qubits

    qc.append(CA(num_qubits, mu), qubits)
    qc.barrier()
    if j > 0:
        qc.append(U(num_qubits), qubits[1:])
        qc.cz(0, j)
        qc.append(U(num_qubits), qubits[1:])
    qc.append(CA(num_qubits, nu), qubits)
    qc.barrier()
    
    return qc


def fixed_ansatz(alpha, num_qubits, n_layers=1):

    qc = QuantumCircuit(num_qubits, name='V')
    qubits = qc.qubits

    for iy in range(num_qubits):
        qc.ry(alpha[iy], qubits[iy])

    layer1 = fixed_layer(alpha[num_qubits:], num_qubits, n_layers)
    qc.append(layer1, qubits)

    return qc


def fixed_layer(alpha, num_qubits, n_layers=1):

    qc = QuantumCircuit(num_qubits, name='V')
    qubits = qc.qubits

    ia = 0
    for _ in range(n_layers):
        for iz in range(0, num_qubits-1, 2):
            qc.cz(iz, iz+1)
            qc.ry(alpha[ia], qubits[iz])
            qc.ry(alpha[ia+1], qubits[iz+1])
            ia += 2

        for iz in range(1, num_qubits-1, 2):
            qc.cz(iz, iz+1)
            qc.ry(alpha[ia], qubits[iz])
            qc.ry(alpha[ia+1], qubits[iz+1])
            ia += 2

    return qc


def CA(num_qubits, idx):
    qc = QuantumCircuit(num_qubits, name='A')
    if idx == 0:
        pass
    elif idx == 1:
        qc.x(1)
        qc.z(2)
    elif idx == 2:
        qc.x(1)

    return qc.to_gate().control()


def U(num_qubits):
    qc = QuantumCircuit(num_qubits, name='U')
    for idx in range(num_qubits):
        qc.h(idx)
    return qc.to_gate()


def construct_circuits(ansatz, num_unitaries):

    circuits = np.empty((ansatz.num_qubits + 1, num_unitaries, num_unitaries),
                        dtype=object)
    for j in range(num_qubits+1):
        for m in range(num_unitaries):
            for n in range(num_unitaries):
                circuits[j, m, n] = construct_term(ansatz, m, n, j=j)
                # draw(circuits[j, m, n][1])
    return circuits


def construct_term(ansatz, mu, nu, j):
    opt = 3
    num_qubits = len(ansatz.qubits)
    qc = QuantumCircuit(num_qubits+1)
    
    qc.append(ansatz, qc.qubits[1:])
    qc.append(local_projector(num_qubits, mu, nu, j=j), qc.qubits)

    hadamard_real = hadamard_test(qc, imaginary=False)
    hadamard_imag = hadamard_test(qc, imaginary=True)
    # draw(hadamard_imag)
    transpiled = qiskit.transpile([hadamard_real, hadamard_imag],
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


def draw(qc):
    fig, ax = plt.subplots(figsize=(15, 10))
    qc.draw('mpl', ax=ax)
    plt.show()


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
    



def global_hamiltonian(A, U):
    n = U.num_qubits 
    b = qi.Statevector(U.to_matrix()[:, 0])
    op = (I^n) - MatrixOp(b.to_operator()).to_pauli_op()

    return  ComposedOp([A.adjoint(), op, A]).to_pauli_op()


if __name__ == '__main__':

    
    backend = qiskit.Aer.get_backend('statevector_simulator',
                                     max_parallel_threads=4,
                                     max_parallel_experiments=4,
                                     precision="single")

    # backend = qiskit.Aer.get_backend('qasm_simulator',
    #                                     max_parallel_threads=8,
    #                                     max_parallel_experiments=16,
    #                                     precision="single")

    num_qubits = 3
    n_shots = 2**11

    # Coefficients of the linear combination A = c0 A0 + c1 A1 + c2 A2
    # A0 = I,
    # A1 = X1*Z2
    # A2 = X1

    c = np.array([1.0, 0.2, 0.2], dtype=complex)

    A_gates = [{},
               {1: library.XGate(), 2: library.ZGate()},
               {1: library.XGate()}]
    
    U_gates = [{i: library.HGate() for i in range(num_qubits)}]

    
    Aop = gates2operator(A_gates, num_qubits, c)
    Uop = gates2operator(U_gates, num_qubits)
    Hop = global_hamiltonian(Aop, Uop)
    
    
    Amat = Aop.to_matrix().real
    bmat = Uop.to_matrix().real[0, :]
    Hmat = Hop.to_matrix()
    
    
    x = np.linalg.solve(Amat, bmat)
    

    n_layers = 6
    alpha = ParameterVector('a', n_layers*4*(num_qubits-1) + num_qubits )
    ansatz = fixed_ansatz(alpha, num_qubits, n_layers=n_layers)

    # ansatz = library.QAOAAnsatz(cost_operator=Hop, reps=6)
    # draw(ansatz.decompose())
    alpha = ansatz.parameters[:]


    print("# Building Circuits...")
    circuits = construct_circuits(ansatz, len(c))
    print("# Circuits Built.")

    n_parameters = len(circuits[0, 0, 0][0].parameters)
    parameters0 = np.random.rand(n_parameters)

        

    test = local_cost(parameters0)

    i = 0

    options = {'maxiter': 150,
                'disp': True}

    cons = ({'type': 'ineq', 'fun': lambda x:  x})
    bounds = [(0, 2*np.pi) for i in range(9)]
    print("# Optimizing")
    result = minimize(local_cost,
                        method='COBYLA',
                    x0=parameters0,
                    options=options,
                    callback=fun_callback)

    
    x_up_to_constant = optimal_state(result.x)
    b_up_to_constant = x_up_to_constant.dot(Amat)
    constant = np.mean(bmat/b_up_to_constant)
    xopt = constant*x_up_to_constant
    bopt = xopt.dot(Amat)
    
    print("# solution ", x)
    print("# approx", xopt)
