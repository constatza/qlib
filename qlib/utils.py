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
from qiskit.quantum_info import Operator

"""
How to apply a unitary gate

unitary = UnitaryGate(matrix, label="U")
circuit.append(unitary, register)

"""


class LinearDecompositionOfUnitaries:
    """Decomposes a matrix A with spectral_norm(A) <=1 to
    a sum of unitary matrices. Stores the appropriate values."""
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.matrix_norm = norm(matrix)
        self.matrix_normalized = self.matrix/self.matrix_norm
        matrices, coeffs = linear_decomposition_of_unitaries(self.matrix_normalized)
        self.decomposition = np.array(matrices)
        self.coeffs = np.array(coeffs)
        self.gates = [UnitaryGate(matrix, label=f'A_{i}') 
                      for i, matrix in enumerate(matrices)]
        self.num_unitaries = len(self.gates)

    def valid_decomposition(self):
        
        unitary_sum = np.sum(self.coeffs[:, None, None] * self.decomposition, axis=0)
        if np.allclose(unitary_sum, self.matrix_normalized):
            return True
        else:
            return False


def unitary_from_hermitian(hermitian: np.ndarray, tol=1e-8):
    """ Decompose a real symmetric matrix to a sum of 
    two unitary operators
    """

    identity = np.eye(N=hermitian.shape[0], dtype=np.complex128)
    
    matrix = identity - hermitian @ hermitian
    sqrt_mat = 1j*sqrtm(matrix)
    F1 = hermitian + sqrt_mat
    F2 = hermitian - sqrt_mat
    return F1, F2


def linear_decomposition_of_unitaries(array: np.ndarray):
    """ Decompose an array with det(array)<=1
    into a linear combination of unitaries
    
    :Input: ndarray with spectral_norm(array)<=1
    :Returns: a list of unitary arrays 
        [F1, F2, F3, F4]
    
    A = B + iC
    
    B = 1/2 (F1 + F2), unitary
    C = 1/2 (F3 + F4), unitary
    
    A = 1/2 (F1 + F2 + iF3 + iF4)
    
    """
    array = array.astype(np.complex128)
    is_hermitian = np.allclose(array, array.conj().T)
    if is_hermitian:
        F1, F2 = unitary_from_hermitian(array)
        return (F1, F2), (.5, .5)
    
    else:
        symmetric_part = 1/2*(array + array.conj().T)
        antisymmetric_part = -1j/2*(array - array.conj().T)
        
        F1, F2 = unitary_from_hermitian(symmetric_part)
        F3, F4 = unitary_from_hermitian(antisymmetric_part)
    
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


def adjoint(matrix: np.ndarray):

    return matrix.conj().T


def normalized(matrix: np.ndarray, return_norm=False):
    matrix_norm = norm(matrix, ord=2)
    return matrix/matrix_norm


def print_time(t1, t2):
    print(f"# Time: {t2-t1:.3f}s")

def draw(qc):
    fig, ax = plt.subplots(figsize=(15, 10))
    qc.draw('mpl', ax=ax)
    plt.show()
