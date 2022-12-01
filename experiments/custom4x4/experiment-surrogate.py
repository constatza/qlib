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
from scipy.optimize import basinhopping


initial_parameter_provider = None
outname = 'last-best'
input_path_physical_parameters = "./input/parameters.in"

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



optimizer = 'bfgs'
optimization_options = {'tol': 1e-4}

model = load_model("./model")


if initial_parameter_provider=='surrogate':
    predictor = SolutionPredictorSurrogate(model, parameters)
elif initial_parameter_provider is None:
    predictor = None

experiment = Experiment(matrices, rhs,
                  optimizer=optimizer,
                 solver=vqls,
                   initial_parameter_predictor=predictor,
                 backend=backend)


logger = FileLogger([name for name in experiment.data.keys()],
                    subdir=outname,
                    dateit=False)


experiment.run(
     logger=logger,
    **optimization_options
               )




#
# cobyla 18s
# powell 70s
