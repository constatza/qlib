#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:17:41 2022

@author: archer
"""

import os
from datetime import datetime
import numpy as np
from scipy.linalg import sqrtm, qr, norm
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.opflow.list_ops import SummedOp

"""
How to apply a unitary gate

unitary = UnitaryGate(matrix, label="U")
circuit.append(unitary, register)

"""


class LinearDecompositionOfUnitaries:
    """Decomposes a matrix A with spectral_norm(A) <=1 to
    a sum of unitary matrices. Stores the appropriate values."""

    def __init__(self, matrix):
        if type(matrix)== np.ndarray:
            self.from_matrix(matrix)
        else:
            self.from_summed_op(matrix)
        self.num_unitaries = len(self.gates)
        self.decomposition = np.array(self.matrices)

    
    def from_summed_op(self, summed_op):
        self.matrix = summed_op.to_matrix()
        self.paulis = summed_op.to_pauli_op()
        try:
            _ = iter(self.paulis)
        except TypeError as te:
            self.paulis = SummedOp([self.paulis, 0*self.paulis])
        self.coeffs = np.array([pauli.coeff for pauli in self.paulis])
        self.coeffs = self.coeffs/norm(self.coeffs)/10
        self.gates = [pauli.to_circuit().to_gate() for pauli in self.paulis]
        self.matrices = [pauli.to_matrix() for pauli in self.paulis]
        self.matrix_norm = norm(self.matrix)
        self.matrix_normalized = self.matrix/self.matrix_norm

    def from_matrix(self, matrix):
        self.matrix = matrix
        self.matrix_norm = norm(self.matrix)
        self.matrix_normalized = self.matrix/self.matrix_norm
        matrices, coeffs = linear_decomposition_of_unitaries(self.matrix_normalized)
        self.matrices = matrices
        self.coeffs = np.array(coeffs)
        self.gates = [UnitaryGate(matrix, label=f'A_{i}')
                      for i, matrix in enumerate(matrices)]

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

    random_array = np.vstack([coeffs.ravel(), np.random.rand(k-1, k)])
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


def print_time(t, msg=''):
    print("# " + msg + f" Elapsed Time: {t:.2f}s")


class FileLogger:

    def __init__(self, filenames, directory='output', dateit=True):

        self.filenames = filenames
        self.dir = directory

        if dateit:
            directory += datetime.today().strftime("%Y%m%d_%H%M")

        try:
            os.makedirs(directory)
        except FileExistsError:
            print(f'{directory} already exists')

        self.directory = directory
        self.paths = [os.path.join(directory, name) for name in filenames]

    def save(self, data_list):
        if len(data_list) != len(self.filenames):
            raise ValueError(
                'Number of inputs must match number of initial filenames')

        for name, array in zip(self.paths, data_list):
            if array is not None:
                with open(name, 'a') as f:
                    array = np.atleast_2d(np.array(array))
                    if array.ndim > 2:
                        dims = array.shape
                        array = array.reshape((-1, dims[1]))
                    np.savetxt(f, array)
