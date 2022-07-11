# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR
from bmapqml.krr_jax import KRR_JAX
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from qml.representations import generate_coulomb_matrix
from bmapqml.fkernels import gaussian_kernel_matrix, gaussian_sym_kernel_matrix
import os, random
import numpy as np

seed = 1
# Replace with path to QM9 directory
QM9_dir = os.environ["DATA"] + "/QM9_formatted"
train_num = 100000
test_num = 1000

size = 29  # maximum number of atoms in a QM9 molecule.

xyz_list = dirs_xyz_list(QM9_dir)
random.seed(seed)
np.random.seed(seed)
random.shuffle(xyz_list)


def get_reps(xyz_list):
    comps = [OML_compound(xyz_file) for xyz_file in xyz_list]
    representations = np.array(
        [
            generate_coulomb_matrix(comp.nuclear_charges, comp.coordinates, size=size)
            for comp in comps
        ]
    )
    return representations


training_reps = get_reps(xyz_list[:train_num])

KRR_model = KRR(kernel_function=gaussian_kernel_matrix)

KRR_model.X_train = training_reps
KRR_model.alphas = np.random.random((len(KRR_model.X_train),))
KRR_model.sigmas = KRR_model.opt_hyperparameter_guess(training_reps[:2000], None)[1:2]

test_reps = get_reps(xyz_list[-test_num:])

KRR_JAX_model = KRR_JAX(KRR_model, "Gaussian")

KRR_JAX_model.prediction_difference(KRR_model, test_reps)
