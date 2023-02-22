import numpy as np
import matplotlib.pyplot as plt
from qlib.visualization import plot_convergence


loss = np.array([
    1.91192e-01,
    1.57178e-01,
    1.21503e-01,
    4.45780e-02,
    3.92787e-02,
    3.01526e-02,
    1.90568e-02,
    9.91253e-03,
    5.90523e-03,
    4.54281e-03,
    2.17355e-03,
    9.79053e-05,
    5.79262e-06,
    1.28937e-06,
    7.44486e-08,
    4.97087e-10,
    6.87950e-13,
    1.30285e-13,
    1.12632e-13,
    9.52016e-14,
    9.40359e-14,
    9.65894e-14,
    9.81992e-14,
    9.54237e-14,
])

fig, ax = plot_convergence(loss, '')
fig.savefig('output/q2_constant/loss.png', dpi=500)
plt.show()
