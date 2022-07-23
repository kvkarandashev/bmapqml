# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.

from bmapqml.hyperparameter_optimization import stochastic_gradient_descent_hyperparam_optimization
from bmapqml.kernels import gaussian_sym_kernel_matrix, gaussian_kernel_matrix
from bmapqml.dataset_processing.qm9_format_specs import Quantity
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from bmapqml.linear_algebra import scipy_cho_solve
from qml.representations import generate_coulomb_matrix
import os, random
import numpy as np

# Replace with path to QM9 directory. 
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

quant_name='HOMO eigenvalue'
seed=1

# Replace with path to QM9 directory
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

train_num=1000
check_num=2000

size=29 # maximum number of atoms in a QM9 molecule.

xyz_list=dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)

def get_quants_comps(xyz_list, quantity):
    quant_vals=np.array([quantity.extract_xyz(xyz_file) for xyz_file in xyz_list])
    comps=[OML_compound(xyz_file) for xyz_file in xyz_list]
    representations=np.array([generate_coulomb_matrix(comp.nuclear_charges, comp.coordinates, size=size) for comp in comps])
    return representations, comps, np.array(quant_vals)

quant=Quantity(quant_name)
training_reps, training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant)

optimized_hyperparams=stochastic_gradient_descent_hyperparam_optimization(training_reps, training_quants, init_param_guess=np.array([1.0, 100.0]), max_stagnating_iterations=8,
                                    randomized_iterator_kwargs={"default_step_magnitude" : 0.05}, sym_kernel_func=gaussian_sym_kernel_matrix, additional_BFGS_iters=8)

sigmas=optimized_hyperparams["sigmas"]
lambda_val=optimized_hyperparams["lambda_val"]

print("Finalized parameters:", sigmas)
print("Finalized lambda:", lambda_val)

K_train=gaussian_sym_kernel_matrix(training_reps, sigmas)
K_train[np.diag_indices_from(K_train)]+=lambda_val
alphas=scipy_cho_solve(K_train, training_quants)
del(K_train)

check_reps, check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant)
K_check=gaussian_kernel_matrix(check_reps, training_reps, sigmas)
predicted_quants=np.dot(K_check, alphas)
MAE=np.mean(np.abs(predicted_quants-check_quants))
print("Quantity: ", quant_name, ", MAE:", MAE)
