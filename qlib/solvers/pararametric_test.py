#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:29:45 2022

@author: archer
"""

import numpy as np
from numpy.linalg import cond
from qlib.solvers.vqls import VQLS, FixedAnsatz, Experiment, RealAmplitudesAnsatz
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qlib.utils import states2qubits
import matplotlib.pyplot as plt
from qiskit import Aer

size = 30

parameter0 = np.linspace(1, 2, size)

parameter1 = np.linspace(2, 3, size)

xx, yy = np.meshgrid(parameter0, parameter1)

x = xx.ravel()
y = yy.ravel()

x = np.linspace(1, 3, 100)
y = np.linspace(2, 4, 100)

matrices = np.array([[-0.5*x**2, x*y],
                     [x*y, 2*y**2 + 1]])



backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=0,
                          max_parallel_experiments=100,
                          precision="single")

matrices = matrices.transpose(2, 0, 1)
matrices = np.block([[3*matrices, 2*matrices],
                      [2*matrices, 4*matrices]])

N = matrices.shape[2]
num_qubits = states2qubits(N)


rhs = np.zeros((N,))
rhs[0] = 1

ansatz = FixedAnsatz(num_qubits=num_qubits, num_layers=3, max_parameters=19)
# ansatz = RealAmplitudesAnsatz(num_qubits=num_qubits, num_layers=1)
num_parameters = ansatz.num_parameters
qc = ansatz.get_circuit()


vqls = VQLS(ansatz=ansatz,
            backend=backend
            )


optimizer = COBYLA(callback=vqls.print_cost)


experiment = Experiment(matrices, rhs,
                        optimizer=optimizer,
                        solver=vqls,
                        backend=backend)


experiment.run(save=False)


optimals = experiment.optimal_parameters
solutions = experiment.solutions



plt.plot(parameter0, np.sin(optimals[:, 0] + optimals[:,1]))

fig, ax = plt.subplots()
ax.plot(x, np.sin(optimals))
ax.plot(x, np.sin(optimals[:, 0] + optimals[:, 1]))
ax.set_xlabel('x1')

# 2D
# ax.set_ylabel('x2')
# zz = optimals.reshape((size, size, -1))
# ax.contourf(xx, yy, np.sin(zz[:, :, 0]))
# ax.contourf(xx, yy, np.sin(np.sum(zz, axis=-1)))

fig, ax = plt.subplots()
theta =np.linspace(0, 4*np.pi)
ax.plot(theta, np.cos(theta/2), theta, -np.sin(theta/2))
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\left|\psi\right>$')
plt.plot()