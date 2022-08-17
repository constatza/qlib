
from fourier import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-q', '--qubits-per-dim',
                    nargs='+',
                    required=False,
                    type=int,
                    default=[4], 
                    dest="num_qubits_per_dimension",
                    help="Number of qubits in each dimension" )

parser.add_argument('-f', '--fourier-per-dim',
                    nargs='+',
                    required=False,
                    type=int,
                    default=[5], 
                    dest="num_fourier_per_dimension",
                    help="Number of generations to calculate" )





num_fourier_per_dimension = np.array([5, 4])
num_qubits_per_dimension = np.array([4, 5])

xpoints, xlower, xupper, xupper_extension = prepare_xpoints([-4, 4], 
                                                            [-4, 4],
                                                            num_qubits_per_dimension)


xmean = 0.5*(xupper+xlower)
               
distribution = [stats.uniform(xlower[0], xupper[0] - xlower[0]),
                 stats.norm(xmean[1], 1)]                                         



prob_dist = [distribution[0].pdf(xpoints[0]), 
                 distribution[1].pdf(xpoints[1])]

prob_dist = np.outer(prob_dist[0], prob_dist[1])


# stats.lognorm(scale=xmean, s=0.25)
# stats.poisson(mu=xmean)
plt.figure()
 
plt.imshow(prob_dist)
#%%
order = 1
def f(x): return (x ) **order
def df(x): return order*(x )**(order-1)



# Extended Fourier
funcs = [f, f]
derivatives = [df, df]
fourier_coeffs_per_dimension = []
for i in range(2):
    
    x = np.linspace(xlower[i], xupper_extension[i], 20000, endpoint=True)
    Func = periodic_extension(f, df,
                              xlower[i], xupper[i], xupper_extension[i])
    period = xupper_extension[i] - xlower[i]
    y = Func(x)


    # Q2
    fourier_coeffs_per_dimension.append(dft(y, x, num_fourier_per_dimension[i]))
    y_complex = idft(fourier_coeffs_per_dimension[-1], x)


backend = Aer.get_backend("statevector_simulator",
                          max_parallel_experiments=4,
                          max_parallel_processes=4)


qae_algorithm = "IAE"
if qae_algorithm == "FAE":
    kwargs = {"delta": 0.05,
              "maxiter": 1000,
              "quantum_instance": backend}
else:
    kwargs = {"alpha": 0.05,
              "epsilon_target": 0.01,
              "quantum_instance": backend}


expected_value_quantum = sum_estimation2d(prob_dist, 
                                        fourier_coeffs_per_dimension,
                                        xlower, xupper, xupper_extension,
                                        qae_algorithm=qae_algorithm,
                                        **kwargs)

#%%
num_samples = 1000000
# samples = distribution.rvs(size=num_samples)
# expected_value_classical = func(samples).sum()/num_samples
# print(expected_value_classical)
fxy = np.outer(funcs[0](xpoints[0]), funcs[1](xpoints[1]))
expected_value_discrete = np.sum(prob_dist*(fxy))
print(expected_value_quantum, expected_value_discrete)


y_real = fourier_from_sines(fourier_coeffs_per_dimension[-1], 2*np.pi/period, x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, y_complex)

plt.show()
