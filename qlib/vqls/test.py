#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""


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



Encoder = np.eye(N, dtype=int)


for i in range(I+1):
init = i * 2**Z + initial[i] 
end = i * 2**Z + final[i]  
Encoder[[init, end]] = Encoder[[end, init]]



 
res = Encoder.dot(basis(4, N))




