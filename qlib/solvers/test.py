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
from qlib.solvers.vqls import VQLS, FixedAnsatz
from qiskit.algorithms.optimizers import SPSA, SciPyOptimizer, CG, COBYLA
from qiskit.circuit.library import RealAmplitudes

backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=4,
                          max_parallel_experiments=20,
                          max_job_size=4,
                          num_shots=1,
                         precision="single")

# backend = qiskit.Aer.get_backend('qasm_simulator',
#                                     max_parallel_threads=8,
#                                     max_parallel_experiments=16,
#                                     precision="single")
num_qubits = 3
size = 2**num_qubits
num_layers = 4
num_shots = 2**11
tol = 1e-3
# np.random.seed(1)

filepath = "../../data/8x8/"
matrices = loadmat(filepath + "stiffnesses.mat")['stiffnessMatricesData'] \
    .transpose(2, 0, 1).astype(np.float64)
solutions = loadmat(filepath + "solutions.mat")['solutionData']
optimals = np.loadtxt("../../experiments/vqls8x8/results/critical/OptimalParameters.out" )


for matrix in matrices:
    matrix[matrix==2e4] = np.max(matrix)

b = np.zeros((8,))
b[3] = -100
b[6] = 100


# matrices = np.array(matrices[0:2, :4, :4])
# b = np.array([1] + [0]*3)

ansatz = FixedAnsatz(num_qubits,
                   num_layers=num_layers)

vqls = VQLS(backend=backend, 
            ansatz=ansatz)

options = {'maxiter': 8000,
           'tol': tol,
           'callback':vqls.print_cost,
           'rhobeg':1e-3}

opt = COBYLA(**options)


x0 = optimals[0, :]


# x0 = np.array([ 1.12883635,  1.79521215,  2.33015176,  2.3826475 , -0.87673251,
#         1.68188146,  0.95878569,  0.97480226,  0.33004086,  0.9000187 ,
#         0.52693897,  0.07575965,  1.95090849, -1.15411322,  0.12271046,
#         1.11621051,  2.38753185,  0.4385254 ,  0.13091519])



for A in matrices[0:1]:
   
    
    x = np.linalg.solve(A, b)

    # ansatz = RealAmplitudes(num_qubits=num_qubits, reps=20)
    
    vqls.A = A
    vqls.b = b

    
    xa = vqls.solve(optimizer=opt, initial_parameters=x0).get_solution(scaled=True)
    ba = xa.dot(A)
    print(xa)