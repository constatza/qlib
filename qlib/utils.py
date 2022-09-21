#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:17:41 2022

@author: archer
"""


import numpy as np
from scipy.linalg import sqrtm, qr, norm
import matplotlib.pyplot as plt
from qiskit.extensions import UnitaryGate

"""
How to apply a unitary gate

unitary = UnitaryGate(matrix, label="U")
circuit.append(unitary, register)

"""


class LinearDecompositionOfUnitaries:
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.matrix_norm = norm(matrix, ord=2)
        self.matrix_normalized = self.matrix/self.matrix_norm
        matrices, coeffs = linear_decomposition_of_unitaries(self.matrix_normalized)
        self.decomposition = np.array(matrices)
        self.coeffs = np.array(coeffs)
        self.gates = list(map(UnitaryGate, matrices))
        self.num_unitaries = len(self.gates)

    def test_sum(self):
        
        return np.sum(self.coeffs * self.decomposition, axis=0)


def unitary_from_hermitian(hermitian: np.ndarray):
    """ Decompose a real symmetric matrix to a sum of 
    two unitary operators
    """

    identity = np.eye(N=hermitian.shape[0])

    term = 1j * sqrtm(identity - hermitian @ hermitian)
    return hermitian + term, hermitian - term


def linear_decomposition_of_unitaries(array: np.ndarray):
    """ Decompose an array with det(array)<=1
    into a linear combination of unitaries
    
    :Input: ndarray with norm(array)<=1
    :Returns: a list of unitary arrays 
        [F1, F2, F3, F4]
    
    A = B + iC
    
    B = 1/2 (F1 + F2), unitary
    C = 1/2 (F3 + F4), unitary
    
    A = 1/2 (F1 + F2 + iF3 + iF4)
    
    """
    real_part = 1/2*(array + array.conj().T)
    imag_part = -1j/2*(array - array.conj().T)

    F1, F2 = unitary_from_hermitian(real_part)
    F3, F4 = unitary_from_hermitian(imag_part)

    return (F1, F2, 1j*F3, 1j*F4), (.5, .5, .5, .5)


def unitary_from_column_vector(coeffs: np.ndarray, *args, **kwargs):
    """Constructs a unitary operator taking the zero state |0>
    to the state with coeffs as amplitudes, using QR decomposition
    """
    k = coeffs.shape[0]

    random_array = np.vstack([coeffs, np.random.rand(k-1, k)])
    unitary, _ = qr(random_array.T)
    is_nonzero = unitary[:, 0].nonzero()[0][0]
    if unitary[is_nonzero, 0] * coeffs[is_nonzero] < 0:
        # assert same sign for operator and coeffs
        unitary *= -1
    return UnitaryGate(unitary, *args, **kwargs)


def states2qubits(num_states: int):
    if num_states==1:
        return 1
    return int(np.ceil(np.log2(num_states) ))


def dagger(matrix: np.ndarray):

    return matrix.conj().T


def normalized(matrix: np.ndarray, return_norm=False):

    matrix_norm = norm(matrix, ord=2)

    return matrix/matrix_norm


def draw(qc):
    fig, ax = plt.subplots(figsize=(15, 10))
    qc.draw('mpl', ax=ax)
    plt.show()
