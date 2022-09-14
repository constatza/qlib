#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:30:22 2022

@author: archer
"""

from lde import *
import matplotlib.pyplot as plt
import numpy as np
import qiskit


taylor_terms = np.arange(1, 3)
M = np.array([[0, 1],
              [-1, 0]])

# M = np.eye(2**num_working_qubits)

# M = np.random.rand(2**num_working_qubits, 2**num_working_qubits)


lcu_coeffs = np.array([.5]*4)
x0 = np.array([1, 0])


backend = qiskit.Aer.get_backend('statevector_simulator',
                                 device='GPU',
                                 max_parallel_threads=20,
                                 max_parallel_experiments=20,
                                 precision="single",
                                 )


time = np.linspace(0.02, 2, 10)


solutions = []
sfs = []
for num_terms in taylor_terms:
    lde = Evolution(M, x0, k=num_terms)
  
    sim = RangeSimulation(lde, backend=backend)

    solutions.append(sim.simulate_range(time, apply_scale=False))
    sfs.append(lde.scale_factor)


fig, ax = plt.subplots()

for i, num_terms in enumerate(taylor_terms):
    x = solutions[i]
    ax.plot(time, x[:, 1], '--', label=f"k={num_terms:d}")


x_classical = sim.simulate_range_exact(time)
ax.plot(time, x_classical[:, 1],  color='g', label="Exact")


ax.set_title("LDE 1-dof system $\ddot{x} + x = 0$")
ax.set_ylabel("$x(t)$")
ax.set_xlabel('$t$')
ax.legend()


# plt.savefig("/home/archer/Documents/phd/presentations/lde/img/solution_per_taylor_terms_1dof_xdot.png",
#             dpi=400,
#             format='png')

plt.show()
