from fourier import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

epsilon = 0.0001
num_fourier = 5
num_qubits = 4

xlower = -3
xupper = 3


func = lambda x: x **2
func_derivative = lambda x: 2*x
real_value = 1

distribution = stats.uniform(xlower, xupper-xlower)
distribution = stats.norm()

xupper_extension = 0.5*(3*xupper - xlower) 

x_piecewise = [xlower, xupper, xupper_extension]

  
# Extended Fourier
Func = periodic_extension(func, func_derivative, xlower, xupper, xupper_extension)
x = np.linspace(xlower, xupper_extension, 20000)
y = Func(x)
coeffs = dft(y, x, num_fourier)
total_error = [] 
with open('error.txt', 'a') as output:
    
    
    num_points = 2**num_qubits
    xpoints = np.linspace(xlower, xupper, num_points) 
    pdf = distribution.pdf(xpoints)
    plt.stem(xpoints, pdf)
    
    
    expected_value_quantum = sum_estimation(pdf, coeffs, x_piecewise, epsilon=epsilon)
    total_error.append((expected_value_quantum - real_value)**2)
    output.write(f"{epsilon} {num_qubits:d} {total_error[-1]:.6f}\n")
    
