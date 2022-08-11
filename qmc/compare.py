#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 10:09:36 2022

@author: archer
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_samples = 1000000


def mse(q):
    return 1/q

number_of_uses = np.logspace(1,6)


fig, ax = plt.subplots()
ax.loglog(number_of_uses, mse(number_of_uses))
ax.loglog(number_of_uses, mse(number_of_uses)**2)


columns = ['qubits','fourier', 'q', 'qmc', 'discrete']  

df = pd.read_csv('./output/eccomas_ex2.dat', names=columns,
                 delimiter='\s+')


real = 1
real = .40235947

df['error_discr'] = (df.qmc - df.discrete)**2
df['error_cont'] = (df.qmc - real)**2
df['perc'] = np.sqrt(df.error_cont)/real

ax = df.plot.scatter(x='q', y='error')#, c='qubits', cmap='RdPu')
ax.set_yscale('log')