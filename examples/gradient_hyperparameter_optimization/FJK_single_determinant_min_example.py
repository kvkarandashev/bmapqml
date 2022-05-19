# Minimal example for calculating energy corrections.
# Note that it requireds presence of QM7bT directory, that can be created with
# procedures from qm7b_t_format_specs.

from molopt.test_utils import dirs_xyz_list
from molopt.linear_algebra import scipy_cho_solve 
from molopt.dataset_processing.qm7b_t_format_specs import Quantity, au_to_kcalmol_mult
import os, random
import numpy as np
from molopt.orb_ml import OML_compound_list_from_xyzs
from molopt.orb_ml.representations import OML_rep_params
from molopt.orb_ml.fkernels import gauss_sep_orb_sym_kernel, gauss_sep_orb_kernel
from molopt.orb_ml.kernels import oml_ensemble_avs_stddevs
from molopt.hyperparameter_optimization import stochastic_gradient_descend_hyperparam_optimization
from molopt.orb_ml.hyperparameter_optimization import Ang_mom_classified_rhf
from molopt.utils import dump2pkl

quant_name='MP2/cc-pVTZ'
seed=1

basis='sto-3g'
max_angular_momentum=1

# Replace with path to QM7b-T directory
QM7bT_dir=os.environ["DATA"]+"/QM7bT_reformatted"

train_num=1000
check_num=2000

xyz_list=dirs_xyz_list(QM7bT_dir)
random.seed(seed)
random.shuffle(xyz_list)

os.environ["MOLOPT_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(orb_atom_rho_comp=0.95, max_angular_momentum=max_angular_momentum)

def get_quants_comps(xyz_list, quantity, oml_representation_parameters):
    comps=OML_compound_list_from_xyzs(xyz_list, calc_type="HF", basis=basis)
    comps.generate_orb_reps(oml_representation_parameters)
    quant_vals=np.array([quantity.extract_xyz(xyz_file)-comp.e_tot*au_to_kcalmol_mult for xyz_file, comp in zip(xyz_list, comps)])
    return comps, quant_vals

quant=Quantity(quant_name)
training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, oml_representation_parameters)

rep_avs, rep_stddevs=oml_ensemble_avs_stddevs(training_comps)

reduced_hyperparam_func=Ang_mom_classified_rhf(use_Gauss=True, rep_params=oml_representation_parameters, stddevs=rep_stddevs)

optimized_hyperparams=stochastic_gradient_descend_hyperparam_optimization(training_comps, training_quants, max_stagnating_iterations=8, num_kfolds=128,
                                    reduced_hyperparam_func=reduced_hyperparam_func, randomized_iterator_kwargs={"default_step_magnitude" : 0.25},
                                    init_red_param_guess=np.zeros((reduced_hyperparam_func.num_reduced_params,)), sym_kernel_func=gauss_sep_orb_sym_kernel)


sigmas=optimized_hyperparams["sigmas"]
lambda_val=optimized_hyperparams["lambda_val"]

print("Finalized parameters:", sigmas)
print("Finalized lambda:", lambda_val)

K_train=gauss_sep_orb_sym_kernel(training_comps, sigmas)
K_train[np.diag_indices_from(K_train)]+=lambda_val
alphas=scipy_cho_solve(K_train, training_quants)
del(K_train)

check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, oml_representation_parameters)
K_check=gauss_sep_orb_kernel(check_comps, training_comps, sigmas)
predicted_quants=np.dot(K_check, alphas)
MAE=np.mean(np.abs(predicted_quants-check_quants))
print("Quantity: ", quant_name, ", MAE:", MAE)
