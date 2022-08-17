#!/usr/bin/env python

from matplotlib.ticker import MaxNLocator
from fourier import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse

plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['savefig.dpi'] = 600

#%%
parser = argparse.ArgumentParser()

parser.add_argument('-q', '--qubits-per-dim',
                    required=False,
                    type=int,
                    default=5, 
                    dest="num_qubits",
                    help="Number of qubits in each dimension" )

parser.add_argument('-f', '--fourier-per-dim',
                    required=False,
                    type=int,
                    default=6, 
                    dest="num_fourier",
                    help="Number of generations to calculate" )


parser.add_argument('--plots', 
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    dest='show')

parser.add_argument('-n', '--name',
                    required=False,
                    type=str,
                    default='', 
                    dest="name",
                    help="name for configuration" )

parser.add_argument('-e', '--error',
                    required=False,
                    type=int,
                    default=2, 
                    dest="error",
                    help="name for configuration" )


# args = parser.parse_args(['-q 6', '-f 4'])
args = parser.parse_args()

#%%
prefix = f"./img/{args.name:s}_q{args.num_qubits}_f{args.num_fourier}_"




noise_model = get_noise_model()


backend = Aer.get_backend("qasm_simulator",
                          basis_gates=noise_model.basis_gates,
                          noise_model=noise_model,
                          max_parallel_experiments=4,
                          max_parallel_processes=4)



num_fourier_per_dimension = np.array([args.num_fourier])
num_qubits_per_dimension = np.array([args.num_qubits])

x_piecewise_per_dimension = np.array([autoextend(1, 5)])



num_dimensions = len(num_qubits_per_dimension)
num_points_per_dimension = 2**num_qubits_per_dimension


xlower, xupper, xupper_extension = x_piecewise_per_dimension.T
xmean = 0.5*(xupper+xlower)


distribution = stats.uniform(xlower[0], xupper[0] - xlower[0])
# distribution = stats.norm(xmean[0])
# distribution = stats.lognorm(scale=xmean, s=0.25)
# distribution = stats.poisson(mu=xmean)


# 1D


xpoints = np.linspace(xlower[0], xupper[0], num_points_per_dimension[0])
prob_dist = distribution.pdf(xpoints)
prob_dist /= prob_dist.sum()

fig, ax = plt.subplots()
ax.stem(xpoints, prob_dist)
ax.set_xlabel('x')
ax.set_ylabel('P(x)')
ax.set_title(f'Discrete probability distribution ($2^n$ points, n={args.num_qubits})')
fig.savefig(prefix+'distribution')

#%%
order = 2
def f(x): return 1/x #(x ) **order
def df(x): return -1/x**2#order*(x )**(order-1)



# Extended Fourier
funcs = [f]
derivatives = [df]


    
x = np.linspace(xlower[0], xupper_extension[0], 20000, endpoint=True)
Func = periodic_extension(f, df,
                          xlower[0], xupper[0], xupper_extension[0])
period = xupper_extension[0] - xlower[0]
y = Func(x)


# Q2
fourier_coeffs_per_dimension = [dft(y, x, num_fourier_per_dimension[0])]
y_complex = idft(fourier_coeffs_per_dimension[0], x)

fig, ax = plt.subplots()
coeffs = fourier_coeffs_per_dimension[0]
ax.plot(np.abs(coeffs), marker='o')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('n')
ax.set_title('Norm of Fourier coefficients')
ax.set_ylabel('$|c_n|$')
fig.savefig(prefix + 'fourier_coeffs')


qae_algorithm = "IAE"
if qae_algorithm == "FAE":
    kwargs = {"delta": 0.05,
              "maxiter": 1000,
              "quantum_instance": backend}
else:
    kwargs = {"alpha": 0.01,
              "epsilon_target": 10**(-args.error),
              "quantum_instance": backend}


expected_value_quantum = sum_estimation(prob_dist, 
                                        fourier_coeffs_per_dimension,
                                        xlower, xupper, xupper_extension,
                                        qae_algorithm=qae_algorithm,
                                        **kwargs)

num_samples = 1000000
samples = distribution.rvs(size=num_samples)
expected_value_classical = f(samples).sum()/num_samples
expected_value_discrete = prob_dist.dot(f(xpoints))






y_real = fourier_from_sines(fourier_coeffs_per_dimension[-1], 2*np.pi/period, x)

fig, ax = plt.subplots()
ax.plot(x[:-1], y[:-1], label='F(x)')
ax.plot(x, y_complex, label=f'Fourier Series, n={args.num_fourier:d}')
ax.vlines([xupper, xlower], np.min(y), np.max(y), colors='red', linestyles='dashed', label='Integration interval')
ax.legend()
ax.set_xlabel('x')
fig.savefig(prefix+"fx")


print(f"Number of qubits: {args.num_qubits:d}")
print(f"Number of Fourier coeffs: {args.num_fourier:d}")
print(f"Accuracy: {args.error}")
print(f"Fourier QMCI: {expected_value_quantum:1.5e}")
print(f"Discrete Sum (Exact): {expected_value_discrete:1.5e}")

if args.show:  
    plt.show()
