#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:25:47 2022

@author: archer
"""

import os
import numpy as np
from keras.models import load_model


from qlib.solvers.vqls import VQLS, FixedAnsatz, Experiment, RealAmplitudesAnsatz, \
    SolutionPredictorSurrogate
from qiskit.algorithms.optimizers import COBYLA, POWELL
from qiskit.circuit.library import RealAmplitudes
from qlib.utils import states2qubits, FileLogger
import matplotlib.pyplot as plt
from qiskit import Aer


experiments_dir = "./results/2022-11-17_21-09"
input_path_physical_parameters = os.path.join(experiments_dir, "parameters.in")

parameters = np.loadtxt(input_path_physical_parameters)

x = parameters[:, 0]
y = parameters[:, 1]



matrices = np.array([[-0.5*x**2, x*y],
                     [x*y, 2*y**2 + 1]])


backend = Aer.get_backend('statevector_simulator',
                          max_parallel_experiments=12,
                          precision="single")

matrices = matrices.transpose(2, 0, 1)

matrices = np.block([[3*matrices, 2*matrices],
                      [2*matrices, 4*matrices]])


N = matrices.shape[2]
num_qubits = states2qubits(N)


rhs = np.zeros((N,))
rhs[0] = 1

ansatz = FixedAnsatz(num_qubits=num_qubits, num_layers=1,
                     max_parameters=3)

num_parameters = ansatz.num_parameters
qc = ansatz.get_circuit()


vqls = VQLS(
            ansatz=ansatz,
            backend=backend
            )


optimizer = COBYLA(callback=vqls.print_cost,
                   tol=1e-6,
                   rhobeg=1e-1)

optimizer = POWELL()

model = load_model("./model")

surrogate_predictor = SolutionPredictorSurrogate(model, parameters)

exp = Experiment(matrices, rhs,
                 optimizer=optimizer,
                 solver=vqls,
                  initial_parameter_predictor=surrogate_predictor,
                 backend=backend)

exp.run()


# cobyla 18s
# powell 70s
