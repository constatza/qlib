from fourier import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


qae_algorithm = "IAE"
if qae_algorithm == "FAE":
    kwargs = {"delta": 0.05,
              "maxiter": 1000,
              "quantum_instance": backend}
else:
    kwargs = {"alpha": 0.05,
              "epsilon_target": 0.01,
              "quantum_instance": backend}


num_qubits_per_dimension = np.array([4, 5])

xpoints, xlower, xupper, xupper_extension = prepare_xpoints([-4, 4], 
                                                            [-4, 4],
                                                            num_qubits_per_dimension)


xmean = 0.5*(xupper+xlower)
               
distribution = [stats.uniform(xlower[0], xupper[0] - xlower[0]),
                 stats.norm(xmean[1], 1)]                                         



prob_dist = [distribution[0].pdf(xpoints[0]), 
                 distribution[1].pdf(xpoints[1])]

pdf = np.outer(prob_dist[0], prob_dist[1])

#distribution = stats.uniform(xlower, xupper-xlower).pdf
# distribution = stats.norm(mean, 1).pdf


pdf_amplitudes, omega, deltaX = prepare_parameters(pdf, xlower, xupper, xupper_extension)


qc = create_cirquit(pdf_amplitudes)


res =  integral(qc, (0, -0), 0, omega, deltaX, xlower,
                   qae_algorithm=qae_algorithm,
                   **kwargs)

classical = np.dot(pdf_normalized, np.cos(n*omega*xpoints))

print(res, classical)
