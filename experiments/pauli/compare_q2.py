
import os
import numpy as np
import matplotlib.pyplot as plt
from qlib.visualization import compare, ExperimentIO

parent_dir = os.path.join(os.path.dirname(os.getcwd()),  "pauli")

io_constant = ExperimentIO(parent_dir, "q2-constant-reduced")
# io_mlp = ExperimentIO(parent_dir, "q3_nn")
io_knn = ExperimentIO(parent_dir, "q2-knn-reduced")

size = 760
iterations_constant = io_constant.loadtxt("NumIterations", ndmin=2).astype(int)[
                      :size
                      ]
iterations_knn = io_knn.loadtxt("NumIterations", ndmin=2).astype(int)[:size]
# iterations_mlp = io_mlp.loadtxt("NumIterations", ndmin=2).astype(int)[:size]


iterations = np.hstack([
    iterations_constant,
     iterations_knn,
     # iterations_mlp,
     ])

mean_knn = np.mean(iterations_knn)
mean_constant = np.mean(iterations_constant)

names = ["Constant", "Nearby"]

fig, ax = compare(
    iterations, names=names,
)
ax.set_ylabel("Iterations")
io_constant.savefig(fig, "iterations-q2")

plt.show()

print("EOF")