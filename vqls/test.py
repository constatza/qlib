#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:56:15 2022

@author: archer
"""


import numpy as np


def basis(i, n):
    x = np.zeros((n,1))
    x[i] = 1
    return x


def num_qubits(number):
    return np.ceil(np.log2(number)).astype(int)
    
    
numbers = np.array([7, 5], dtype=int)

I = num_qubits(numbers.shape[0])
Z = num_qubits(numbers.max()+1)

numbers = np.pad(numbers, (0, 2**I-len(numbers)), 
                 'constant', constant_values=(0,0)) 


initial = np.zeros((2**I), dtype=int)
final = initial^numbers

N =  2**(I + Z)



Encoder = np.eye(N, dtype=int)


for i in range(I+1):
    init = i * 2**Z + initial[i] 
    end = i * 2**Z + final[i]  
    Encoder[[init, end]] = Encoder[[end, init]]
    
    

 
res = Encoder.dot(basis(4, N))




