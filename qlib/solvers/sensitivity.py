#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:45:50 2022

@author: archer
"""

import numpy as np

from qiskit import Aer
from qlib.utils import states2qubits
from qlib.solvers.vqls import VQLS, FixedAnsatz
from qiskit.circuit.library import RealAmplitudes
import matplotlib.pyplot as plt


ansatz = FixedAnsatz(num_qubits=3, num_layers=2)
qc = ansatz.get_circuit()

a = np.ones(qc.num_parameters)


tau = np.linspace(-2*np.pi, 2*np.pi, 2000)


xs = []
for ai in tau:
    a[0] = np.sin(ai)
    a[1] = 2*ai
    a[2] = ai**2
    a[3] = 4 * ai
    x = ansatz.get_state(a)
    xs.append(x)




plt.plot(tau, xs)
plt.show()

    