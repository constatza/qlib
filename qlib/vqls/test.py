#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
from vqls import *

backend = Aer.get_backend('statevector_simulator',
                         max_parallel_threads=4,
                         max_parallel_experiments=4,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
size = 4
num_layers = 1
num_shots = 2**11

b = np.ones(size)
A = np.random.rand(size, size)


ansatz = FixedAnsatz(states2qubits(size), num_layers=num_layers)
parameters0 = np.random.rand(ansatz.num_parameters)

vqls = VQLS(A, b, ansatz)


x = np.linalg.solve(A, b)

b_compl = b
b_compl[::2] *= -1
# ansatz = library.QAOAAnsatz(cost_operator=Hop, reps=6)
# draw(ansatz.decompose())




options = {'maxiter': 50,
    'disp': True}

cons = ({'type': 'ineq', 'fun': lambda x:  x})
bounds = [(0, 2*np.pi) for i in range(9)]
print("# Optimizing")
result = minimize(vqls.local_cost,
            method='COBYLA',
            x0=parameters0,
            options=options,
            callback=vqls.fun_callback)


x_up_to_constant = vqls.optimal_state(result.x)
b_up_to_constant = x_up_to_constant.dot(A)
constant = np.mean(b/b_up_to_constant)
xopt = constant*x_up_to_constant
bopt = xopt.dot(A)

print("# solution ", x)
print("# approx", xopt)






