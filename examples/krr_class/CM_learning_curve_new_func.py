# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR, learning_curve
from bmapqml.dataset_processing.qm9_format_specs import Quantity
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from qml.representations import generate_coulomb_matrix
import os, random
import numpy as np

# Replace with path to QM9 directory.
QM9_dir = os.environ["DATA"] + "/QM9_formatted"

quant_name = "HOMO-LUMO gap"
seed = 1

# Replace with path to QM9 directory
QM9_dir = os.environ["DATA"] + "/QM9_formatted"

train_nums = [1000, 2000, 4000, 8000]
bigger_train_subset = 32000
hyperparam_opt_num = 2000
check_num = 8000

size = 29  # maximum number of atoms in a QM9 molecule.

xyz_list = dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)


def get_quants_comps(xyz_list, quantity):
    quant_vals = np.array([quantity.extract_xyz(xyz_file) for xyz_file in xyz_list])
    comps = [OML_compound(xyz_file) for xyz_file in xyz_list]
    representations = np.array(
        [
            generate_coulomb_matrix(comp.nuclear_charges, comp.coordinates, size=size)
            for comp in comps
        ]
    )
    return representations, comps, np.array(quant_vals)


quant = Quantity(quant_name)
all_training_reps, all_training_comps, all_training_quants = get_quants_comps(
    xyz_list[:bigger_train_subset], quant
)

KRR_model = KRR(
    hyperparam_opt_kwargs={
        "max_stagnating_iterations": 1,
        "randomized_iterator_kwargs": {"default_step_magnitude": 0.05},
    },
)

KRR_model.optimize_hyperparameters(
    all_training_reps[:hyperparam_opt_num], all_training_quants[:hyperparam_opt_num]
)

check_reps, check_comps, check_quants = get_quants_comps(xyz_list[-check_num:], quant)

MAEs = learning_curve(
    KRR_model,
    all_training_reps,
    all_training_quants,
    check_reps,
    check_quants,
    train_nums,
)

print("Quantity: ", quant_name, ", Ntrain and MAEs:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, MAE_line)

print("Compact form of the lc:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, np.mean(MAE_line), np.std(MAE_line))
