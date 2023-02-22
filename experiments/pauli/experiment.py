import os
import sys
import numpy as np
from keras.models import load_model
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit import Aer, QuantumCircuit
from qlib.utils import FileLogger
from qlib.solvers.vqls import Experiment, FixedAnsatz, VQLS, SolutionPredictorSurrogate, SolutionPredictorConstant, SolutionPredictorRandom


print(sys.argv[1])
# %% INPUT
INITIAL_PARAMETER_PROVIDER = 'random'
NUM_QUBITS = int(sys.argv[1])
OPTIMIZER = 'BFGS'
OPTIMIZATION_OPTIONS = {
    'tol': 1e-9,
     'options': {'maxiter': 10000, }
}

save_dir = f'q{NUM_QUBITS:d}-' + str(INITIAL_PARAMETER_PROVIDER) + '-'

NUM_POINTS = int(sys.argv[2])

x_ = np.linspace(0., 1., NUM_POINTS)
y_ = np.linspace(1., 2., NUM_POINTS)
z_ = np.linspace(3., 4., NUM_POINTS)
backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=12,
                            max_parallel_experiments=12,
                            nshots=1)


parameter_save_path = os.path.join('input', save_dir + 'params')

# %%
N = 2**NUM_QUBITS
xx, yy, zz = np.meshgrid(x_, y_, z_, indexing='ij')
x = xx.ravel()
y = yy.ravel()
z = zz.ravel()
np.save(parameter_save_path, np.vstack([x, y, z]).T)

if NUM_QUBITS == 3:
    parameters = np.array([[x**3, x**2*y, x*y**2],
                           [z*x**3, x*y, z**2 + x],
                           [z+y**2, z*x + y, y**3 + x]])
elif NUM_QUBITS == 2:
    parameters = np.array([[np.cos(x + y), np.sin(y)],
                           [-np.sin(z), np.cos(x**2 + z)]])
    

parameters = 1/2*(parameters + parameters.transpose(1, 0, 2))

eye = Operator(np.eye(N))
XZ = Operator(Pauli('XZ'))

matrices = []
for n in range(parameters.shape[-1]):
    op = eye
    params = parameters[:, :, n]
    norm = np.linalg.norm(params)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                term = params[i, j]*eye.compose(XZ, qargs=[i, j])
                op += term/norm

    matrices.append(op.data)

matrices = np.array(matrices)

num_samples = matrices.shape[0]
rhs = np.zeros((N,))
rhs[0] = 1

# %%
ansatz = FixedAnsatz(num_qubits=NUM_QUBITS,
                     num_layers=2,
                     max_parameters=2**NUM_QUBITS -1)

if NUM_QUBITS==2:
    params = ansatz.parameters
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cnot(0, 1)
    qc.ry(params[2], 0)
    ansatz.circuit = qc



if INITIAL_PARAMETER_PROVIDER == 'mlp':
    model = load_model('model0')
    predictor = SolutionPredictorSurrogate(
        model,
        parameters,
        training_size=int(0.1*num_samples),
    )
elif INITIAL_PARAMETER_PROVIDER=='constant':
    predictor = SolutionPredictorConstant(ansatz.num_parameters, 1)
elif INITIAL_PARAMETER_PROVIDER == 'random':
    predictor = SolutionPredictorRandom(ansatz.num_parameters)
elif INITIAL_PARAMETER_PROVIDER is None:
    predictor = None

experiment = Experiment(matrices, rhs, ansatz,
                        optimizer=OPTIMIZER,
                        initial_parameter_predictor=predictor,
                        save_path=os.path.join('output', save_dir),
                        dateit=True,
                        backend=backend,
                        )


experiment.run(**OPTIMIZATION_OPTIONS)
