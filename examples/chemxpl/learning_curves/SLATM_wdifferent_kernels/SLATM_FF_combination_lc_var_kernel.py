# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR, learning_curve
from bmapqml.fkernels import (
    gaussian_kernel_matrix,
    gaussian_sym_kernel_matrix,
    laplacian_kernel_matrix,
    laplacian_sym_kernel_matrix,
)
from bmapqml.hyperparameter_optimization import max_Laplace_dist, max_Gauss_dist
from bmapqml.dataset_processing.qm9_format_specs import Quantity
from bmapqml.test_utils import dirs_xyz_list, timestamp
from bmapqml.orb_ml import OML_compound
from qml.representations import get_slatm_mbtypes, generate_slatm
import os, random, sys
import numpy as np

# Choose which kernel to use and which quantity to consider
quant_name = sys.argv[1].replace("_", " ")
kernel_type = sys.argv[2]
ff_type = sys.argv[3]

if kernel_type == "Laplacian":
    kernel_matrix = laplacian_kernel_matrix
    sym_kernel_matrix = laplacian_sym_kernel_matrix
    init_sigma_guess_func = max_Laplace_dist
else:
    kernel_matrix = gaussian_kernel_matrix
    sym_kernel_matrix = gaussian_sym_kernel_matrix
    init_sigma_guess_func = max_Gauss_dist

# Replace with path to QM9 directory.
QM9_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"

seed = 1

# Replace with path to QM9 directory
FF_QM9_dir = os.environ["DATA"] + "/QM9_filtered/"+ff_type+"_xyzs"

train_nums = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
bigger_train_subset = 100000
hyperparam_opt_num = 32000
check_num = 10000

size = 29  # maximum number of atoms in a QM9 molecule.

ff_xyz_list = dirs_xyz_list(FF_QM9_dir)
random.seed(seed)
random.shuffle(ff_xyz_list)


def get_quants_comps(ff_xyz_list, quantity):
    quant_vals = []
    for ff_xyz_file in ff_xyz_list:
        true_xyz_file = QM9_dir + "/" + os.path.basename(ff_xyz_file)
        quant_vals.append(quantity.extract_xyz(true_xyz_file))

    quant_vals = np.array(quant_vals)
    comps = [OML_compound(ff_xyz_file) for ff_xyz_file in ff_xyz_list]

    ncharges = np.zeros((len(ff_xyz_list), size), dtype=int)
    comps = []

    for i, ff_xyz_file in enumerate(ff_xyz_list):
        comp = OML_compound(ff_xyz_file)
        ncharges[i, : len(comp.nuclear_charges)] = comp.nuclear_charges[:]
        comps.append(comp)

    mbtypes = get_slatm_mbtypes(ncharges)

    representations = np.array(
        [
            generate_slatm(comp.coordinates, comp.nuclear_charges, mbtypes)
            for comp in comps
        ]
    )
    return representations, comps, np.array(quant_vals)


quant = Quantity(quant_name)
all_training_reps, all_training_comps, all_training_quants = get_quants_comps(
    ff_xyz_list[:bigger_train_subset], quant
)

KRR_model = KRR(
    init_sigma_guess_func=init_sigma_guess_func,
    kernel_function=kernel_matrix,
    sym_kernel_function=sym_kernel_matrix,
)

timestamp("Optimizing hyperparameters.")
KRR_model.optimize_hyperparameters(
    all_training_reps[:hyperparam_opt_num], all_training_quants[:hyperparam_opt_num]
)
timestamp("Finished optimizing hyperparameters.")

check_reps, check_comps, check_quants = get_quants_comps(
    ff_xyz_list[-check_num:], quant
)

timestamp("Building learning curve.")
MAEs = learning_curve(
    KRR_model,
    all_training_reps,
    all_training_quants,
    check_reps,
    check_quants,
    train_nums,
    model_fit_reusable=True,
)
timestamp("Finished building learning curve.")

KRR_model.save("krr_model.pkl")

print("Quantity: ", quant_name, ", Ntrain and MAEs:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, MAE_line)

print("Compact form of the lc:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, np.mean(MAE_line), np.std(MAE_line))

print(
    "Double-checked MAE of saved model:",
    np.mean(np.abs(KRR_model.predict(check_reps) - check_quants)),
)
