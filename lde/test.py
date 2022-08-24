#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:36:50 2022

@author: archer
"""

from lde import *
import matplotlib.pyplot as plt
import numpy as np
import qiskit



taylor_terms = 2

M = np.array([[0, -1],
          [1, 0]])

# M = np.eye(2**num_working_qubits)

# M = np.random.rand(2**num_working_qubits, 2**num_working_qubits)



 
  
lcu_coeffs = np.array([.5]*4)
x0 = -np.array([1, 0.5])
  


backend = qiskit.Aer.get_backend('statevector_simulator',
                           device='GPU',
                              max_parallel_threads=4,
                              max_parallel_experiments=4,
                               precision="single",
                              )


time = np.linspace(0.01, 3)


lde = UnitaryEvolution(M, x0, k=taylor_terms)

sim = RangeSimulation(lde, backend=backend)

x = sim.simulate_range(time)
x_classical = sim.simulate_range_exact(time)

lines = plt.plot(time, x, time, x_classical)
plt.title(f"Taylor terms k={taylor_terms:d}")
plt.xlabel('t')

plt.legend(iter(lines), ('$\dot x$', '$x$', 'exact $\dot x$', 'exact $x$'))

