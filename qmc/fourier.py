#!/usr/bin/env python
# coding: utf-8

# # Quantum Monte Carlo


from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import AncillaRegister, Parameter, ParameterVector
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem, FasterAmplitudeEstimation
from qiskit.tools.visualization import circuit_drawer
from qiskit import Aer
import qiskit.providers.aer.noise as noise

# Useful additional packages
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


backend = Aer.get_backend('qasm_simulator',
                          max_parallel_experiments=4,
                          max_parallel_processes=4)


def draw(qc):
    fig, ax = plt.subplots(figsize=(10, 5))
    qc.draw('mpl', ax=ax)
    return fig


def prepare_xpoints(x0, x1, num_qubits_per_dimension):
    
    x_piecewise_per_dimension = np.array([autoextend(x0[0], x0[1]), 
                                          autoextend(x1[0], x1[1])])

    
    num_points_per_dimension = 2**num_qubits_per_dimension.astype(int)


    xlower, xupper, xupper_extension = x_piecewise_per_dimension.T
  


    xpoints = [np.linspace(xlower[0], xupper[0], num_points_per_dimension[0]),
               np.linspace(xlower[1], xupper[1], num_points_per_dimension[1])]
               

    return xpoints, np.array(xlower), np.array(xupper), np.array(xupper_extension)


def create_cirquit(amplitudes, backend=backend):
    
    
    total_qubits = int(np.log2(amplitudes.size))
    registers = []
    
    num_qubits_per_dimension = get_num_qubits_from_shape(amplitudes)


    if len(num_qubits_per_dimension)==1:
        num_thetas = 1
        num_qubits_per_dimension = [total_qubits]
        registers.append(QuantumRegister(total_qubits, name="x"))
    else:
        num_thetas = len(num_qubits_per_dimension)
        for i, num_qubits in enumerate(num_qubits_per_dimension):
            registers.append(QuantumRegister(num_qubits, name=f"x{i:d}"))
            
    angle_init = Parameter('α')
    thetas = ParameterVector('θ', length=num_thetas)

    ancilla = QuantumRegister(1, name='ancilla')

    qc = QuantumCircuit(*registers, ancilla)
    qubits = qc.qubits

    qc.initialize(amplitudes.flatten(), qubits[:total_qubits])
    qc.ry(angle_init, ancilla[0])

    for j, (reg, num_qubits) in enumerate(zip(registers, num_qubits_per_dimension)):
        theta = thetas.params[j]
        crots = QuantumCircuit(reg, ancilla, name=f"CY{j:d}")

        for i in range(num_qubits):
            crots.cry(2**i * theta, reg[i], ancilla[0])

        qc.append(crots, [*reg, *ancilla])
    draw(qc.decompose())
    return transpile(qc, backend, optimization_level=3)


def amplitude_estimation(qc, angle_init, thetas, qae_algorithm="FAE", **kwargs):
    # # Quantum Amplitude Estimation using IAE
    par_angle_init = qc.parameters[0]

    qc = qc.bind_parameters({par_angle_init: angle_init})
    
    qc = qc.bind_parameters(dict(zip(qc.parameters, thetas)))
    # construct amplitude estimation
    problem = EstimationProblem(state_preparation=qc,
                                objective_qubits=[qc.num_qubits-1])

    if qae_algorithm == "FAE":
        QAE = FasterAmplitudeEstimation(**kwargs)
    else:
        QAE = IterativeAmplitudeEstimation(**kwargs)

    result = QAE.estimate(problem)
    return result.estimation


def fft_coeffs(y, terms, return_complex=True):
    complex_coeffs = np.fft.rfft(y, len(y))/len(y)
    np.put(complex_coeffs, range(terms+1, len(complex_coeffs)), 0.0)
    complex_coeffs = complex_coeffs[:terms+1]

    if return_complex:
        return complex_coeffs
    else:
        complex_coeffs *= 2
        return complex_coeffs.real[0]/2, complex_coeffs.real[1:-1], -complex_coeffs.imag[1:-1]


def cubic_base(x):
    return x**np.arange(4)


def cubic_base_derivative(x):
    return np.arange(4)*x**np.array([0, 0, 1, 2])


def periodic_extension(func, func_derivative, xl, xu, xuu, extension_only=False):
    xl = float(xl)
    xu = float(xu)
    xuu = float(xuu)
    system_matrix = np.vstack([cubic_base(xuu),
                               cubic_base_derivative(xuu),
                               cubic_base(xu),
                               cubic_base_derivative(xu)])

    constraints = np.array(
        [func(xl), func_derivative(xl), func(xu), func_derivative(xu)])
    cubic_coeffs = np.linalg.solve(system_matrix, constraints)

    def cubic_value(x):
        return cubic_coeffs.dot(cubic_base(x))
    
    cubic_extension = np.vectorize(cubic_value)
    
    if extension_only:
        return cubic_extension
    else:
        return (lambda x: np.piecewise(x, [x <= xu, x > xu], [func, cubic_extension]))


def integral(qc, index_combo, beta, omega, deltaX, xlower, qae_algorithm="FAE", **kwargs):
    if type(index_combo) is int:
        index_combo = [index_combo]
    index_combo = np.array(index_combo)
    angle_init = index_combo.dot(omega * xlower) - beta
    theta = index_combo * omega * deltaX
    phase_good = amplitude_estimation(qc, angle_init, theta, qae_algorithm=qae_algorithm, **kwargs)
    return 1 - 2 * phase_good


def sum_estimation(pdf, fourier_coeffs_per_dimension, 
                   xlower, xupper, xupper_extension, 
                   qae_algorithm="FAE", 
                   backend=backend, **kwargs):
    
    
    pdf_amplitudes, omega, deltaX = prepare_parameters(pdf, xlower, xupper, xupper_extension)
    qc = create_cirquit(pdf_amplitudes)

    fourier_coeffs = fourier_coeffs_per_dimension[0]
 
    
    for n in range(1, len(fourier_coeffs_per_dimension[0])):
        
        
        cos_sum = integral(qc, n, 0, omega, deltaX, xlower,
                           qae_algorithm=qae_algorithm,
                           **kwargs)

        sin_sum = integral(qc, n, np.pi/2, omega, deltaX, xlower,
                           qae_algorithm=qae_algorithm,
                           **kwargs)
        
        
        fourier_coeffs.real[n] *= 2*cos_sum
        fourier_coeffs.imag[n] *= 2*sin_sum
    s = fourier_coeffs.real.sum() - fourier_coeffs.imag.sum()

    return s


def sum_estimation2d(pdf, fourier_coeffs_per_dimension, 
                     xlower, xupper, xupper_extension, 
                   qae_algorithm="FAE", 
                   backend=backend, **kwargs):
    
    
   
    pdf_amplitudes, omega, deltaX = prepare_parameters(pdf, xlower, xupper, xupper_extension)
    qc = create_cirquit(pdf_amplitudes)
    
    # real formula requires 2*ci except for real[0]
    
    a1 = 2*fourier_coeffs_per_dimension[0].real
    b1 = 2*fourier_coeffs_per_dimension[0].imag
    a2 = 2*fourier_coeffs_per_dimension[1].real
    b2 = 2*fourier_coeffs_per_dimension[1].imag
    
    a1[0] /= 2
    a2[0] /= 2
    
    c1 = np.outer(a1, a2)
    c2 = np.outer(b1, b2)
    s1 = np.outer(a1, b2)
    s2 = np.outer(b1, b2)
    
    
    cos_sum_minus = np.zeros(c1.shape)
    cos_sum_plus = np.zeros(c1.shape)
    sin_sum_plus = np.zeros(c1.shape)
    sin_sum_minus = np.zeros(c1.shape)
    
    for n in range(fourier_coeffs_per_dimension[0].shape[0]):
        for m in range(fourier_coeffs_per_dimension[1].shape[0]):
        
        
            cos_sum_minus[n, m] = integral(qc, (n, -m), 0, omega, deltaX, xlower,
                               qae_algorithm=qae_algorithm,
                               **kwargs)
            
            cos_sum_plus[n, m] = integral(qc, (n, m), 0, omega, deltaX, xlower,
                               qae_algorithm=qae_algorithm,
                               **kwargs)
            
            sin_sum_plus[n, m] = integral(qc, (n, m), np.pi/2, omega, deltaX, xlower,
                               qae_algorithm=qae_algorithm,
                               **kwargs)
            
            sin_sum_minus[n, m] = integral(qc, (n, -m), np.pi/2, omega, deltaX, xlower,
                               qae_algorithm=qae_algorithm,
                           **kwargs)
        
        
       
    total = 0.5*(c1 + c2) * cos_sum_minus
    total += 0.5*(c1 - c2) * cos_sum_plus
    total += 0.5*(s1 + s2) * sin_sum_plus
    total += 0.5*(s1 - s2) * sin_sum_plus
    
    return total.sum()


def prepare_parameters(pdf, xlower, xupper, xupper_extension):

     num_points_per_dim = np.array(pdf.shape, dtype=int)
     period = xupper_extension - xlower
     omega = 2*np.pi / period
     deltaX = (xupper-xlower)/(num_points_per_dim - 1)
    
     pdf_normalized = pdf/pdf.sum()
     pdf_amplitudes = np.sqrt(pdf_normalized)

     return pdf_amplitudes, omega, deltaX




def plot_fourier_from_fft(fourier_coeffs, x):
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.fft.irfft(fourier_coeffs, len(x))  # *len(x)
    ax.plot(x, y)
    return ax


def fourier_from_sines(coeffs, omega, x):
    coeffs *= 2
    a0 = coeffs.real[0]/2
    a = coeffs.real[1:]
    b = coeffs.imag[1:]
    y = a0
    for n in range(a.shape[0]):
        phase = omega*x*(n+1)
        y += a[n] * np.cos(phase) + b[n] * np.sin(phase)
    return y


def dft(y, x, nterms):
    yr = recast_for_dft(y)
    period = x[-1] - x[0]
    coeffs = [yr.mean()]
    for n in range(1, nterms):
        cn = yr*np.exp(-1j*2*n*np.pi*x/period)
        coeffs.append(cn.mean())
    return np.array(coeffs)


def idft(complex_coeffs, x):
    period = x[-1] - x[0]
    fx = complex_coeffs[0]*np.ones_like(x)
    for i in range(1, complex_coeffs.size):
        fx += 2*complex_coeffs[i]*np.exp(1j*2*i*np.pi*x/period)
    return fx.real


def recast_for_dft(array):
    array[0] += array[-1]
    array[0] /= 2
    array[-1] = 0
    return array


def autoextend(x0, x1, coeff=0.5):
    return np.array([x0, x1, x1 + coeff*(x1 - x0)])

def get_noise_model():
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate
    
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)
    
    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    
    # Get basis gates from noise model

    return noise_model


def to_num_qubits(arraylike):
    return np.ceil(np.log2(arraylike)).astype(int)


def get_num_qubits_from_shape(arraylike):
    return to_num_qubits(np.array(arraylike).shape)


if __name__ == '__main__':
    pass
