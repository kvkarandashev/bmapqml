# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR, learning_curve
from bmapqml.fkernels import (
    gaussian_kernel_matrix,
    gaussian_sym_kernel_matrix,
    laplacian_kernel_matrix,
    laplacian_sym_kernel_matrix,
)
from bmapqml.hyperparameter_optimization import max_Laplace_dist, max_Gauss_dist
from bmapqml.dataset_processing.qm9_format_specs import Quantity, read_SMILES
from bmapqml.test_utils import dirs_xyz_list, timestamp
from bmapqml.chemxpl.rdkit_descriptors import get_all_FP
import os, random, sys
import numpy as np

# Choose which kernel to use and which quantity to consider
quant_name = sys.argv[1].replace("_", " ")
kernel_type = sys.argv[2]
radius_parameter = int(sys.argv[3])
useFeatures = sys.argv[4] == "True"

fp_type = "MorganFingerprint"

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

train_nums = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
bigger_train_subset = 100000
hyperparam_opt_num = 32000
check_num = 10000

size = 29  # maximum number of atoms in a QM9 molecule.

xyz_list = dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)


def get_quants_SMILES(xyz_list, quantity):
    quant_vals = []
    SMILES_list = []
    representations = []
    for xyz_file in xyz_list:
        quant_vals.append(quantity.extract_xyz(xyz_file))
        cur_SMILES = read_SMILES(xyz_file)
        SMILES_list.append(cur_SMILES)

    representations = get_all_FP(
        SMILES_list,
        fp_type,
        radius=radius_parameter,
        nBits=8192,
        useFeatures=useFeatures,
    )

    return np.array(representations), SMILES_list, np.array(quant_vals)


quant = Quantity(quant_name)
all_training_reps, all_training_SMILES, all_training_quants = get_quants_SMILES(
    xyz_list[:bigger_train_subset], quant
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

check_reps, check_SMILES, check_quants = get_quants_SMILES(xyz_list[-check_num:], quant)

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
