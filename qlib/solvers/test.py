#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""
import numpy as np
from scipy.io import loadmat
from qiskit import Aer
from qlib.utils import states2qubits
from qlib.solvers.vqls import VQLS, FixedAnsatz, RealAmplitudesAnsatz
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, CG, COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer.backends.aer_simulator import AerSimulator


backend = Aer.get_backend('statevector_simulator',
                          max_parallel_experiments=4,

                          num_shots=1,
                         precision="single")

num_qubits = 3
num_layers = 2

size = 2**num_qubits
num_shots = 2**11
tol = 1e-8
# np.random.seed(1)

filepath = "../../data/8x8/"
matrices = loadmat(filepath + "stiffnesses.mat")['stiffnessMatricesData'] \
    .transpose(2, 0, 1).astype(np.float64)
parameters = loadmat(filepath + "parameters.mat")['parameterData'].T
optimals = np.loadtxt("../../experiments/vqls8x8/results/continuous/OptimalParameters_2022-11-02_17-19.txt")


for matrix in matrices:
    matrix[matrix==2e4] = np.max(matrix)


b = np.zeros((8,))
b[3] = -100
b[6] = 100


# matrices = np.array(matrices[0:2, :4, :4])
# b = np.array([1] + [0]*3)
# N = 2**num_qubits
# matrices = np.random.rand(2, N, N)
# matrices = 0.5*(matrices + matrices.transpose(0, 2, 1))
# b = np.random.rand(N,1)


ansatz = FixedAnsatz(num_qubits,
                   num_layers=num_layers)

# ansatz = RealAmplitudesAnsatz(num_qubits=num_qubits,
#                    num_layers=num_layers)

qc = ansatz.get_circuit()
print(qc)

vqls = VQLS(backend=backend,
            ansatz=ansatz)

options = {'maxiter': 8000,
           'tol': tol,
           'callback':vqls.print_cost,
           'rhobeg':1e-5}

opt = COBYLA(**options)






from keras.models import load_model


model = load_model("/home/archer/code/quantum/qlib/ml/model0")

x0 = model.predict(parameters[0:1,:])

for A in matrices[0:1]:

    x = np.linalg.solve(A, b)

    vqls.A = A
    vqls.b = b

    xa = vqls.solve(optimizer=opt,
                    initial_parameters=x0,
                    rhobeg=1e-4).get_solution(scaled=True)
    ba = xa.dot(A)
    print(xa)
