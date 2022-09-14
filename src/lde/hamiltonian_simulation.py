#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:56:25 2022

@author: archer
"""
import matplotlib.pyplot as plt
from qiskit.opflow import X, Y, Z, I, PauliTrotterEvolution
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
import numpy as np 
from qiskit.opflow.primitive_ops import PrimitiveOp, PauliSumOp
from qiskit.extensions import HamiltonianGate
from qiskit import Aer, execute
from scipy.linalg import expm


random = np.array([[1, 0], 
                   [2, -1]], dtype=complex)

skew = np.array([[0, 0],
                 [1, 0]], dtype=complex)

matrix = 1j*(np.kron(skew, random) + -np.kron(skew.T.conjugate(), random.T.conjugate()))


hamiltonian = PrimitiveOp(Operator(data=matrix))

# hamiltonian = 3*(X^X^Z) - 1*(Z^X^Z)
evo_time = Parameter('t')
evolution_op = (evo_time*hamiltonian).exp_i()
print(evolution_op)
# into circuit
num_time_slices = 3
trotterized_op = PauliTrotterEvolution(
                    trotter_mode='trotter',
                    reps=num_time_slices).convert(evolution_op)



# fig, ax = plt.subplots()
# trotterized_op.to_circuit().bind_parameters({evo_time:2}).decompose(reps=1).draw('mpl', ax=ax)
# plt.show()

qubits = QuantumRegister(2)
qc = QuantumCircuit(qubits)
hg = HamiltonianGate(matrix, evo_time)
qc.append(hg, qubits)


fig, ax = plt.subplots()
qc.draw('mpl', ax=ax)
plt.show()

backend = Aer.get_backend('statevector_simulator')

qcs = []

time = np.linspace(0, 20, 100)
for i, t in enumerate(time):

    qcs.append(qc.assign_parameters({evo_time: t}))
    
result = execute(qcs, backend).result()

solutions = []
exact = []
for i, t in enumerate(time):
    
    x = result.get_statevector(i).data.real[2:]
    solutions.append(x)
    exact.append( expm(random*t) @ np.array([0, 0])
   
solutions = np.array(solutions)
plt.plot(time, solutions[:, 0])