# From KRR pickle files created in examples/chemxpl/learning_curves create
# Move those pickle files to
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.minimized_functions import (
    SLATM_MMFF_based_model,
    LinearCombination,
)
import os
from bmapqml.utils import loadpkl, dump2pkl
from bmapqml.dataset_processing.qm9_format_specs import Quantity
import numpy as np

QM9_xyz_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"
MMFF_xyz_dir = os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs"

pkl_store_dir = "/store/common/konst/chemxpl_related"

mmff_xyz_list = dirs_xyz_list(MMFF_xyz_dir)


def quant_QM9_stddev(quant_name):
    quantity = Quantity(quant_name)
    quant_vals = []
    for mmff_xyz_file in mmff_xyz_list:
        true_xyz_file = QM9_xyz_dir + "/" + os.path.basename(mmff_xyz_file)
        quant_vals.append(quantity.extract_xyz(true_xyz_file))
    return np.std(np.array(quant_vals))


quant_names_list = [
    ("Dipole moment", "HOMO-LUMO gap"),
    ("HOMO-LUMO gap", "Normalized atomization energy"),
]

quant_signs_list = [(-1, -1), (1, 1)]

min_func_keywords = ["electrolyte", "chromophore"]

min_functions = []

for quant_names, quant_signs in zip(quant_names_list, quant_signs_list):
    SLATM_models = []
    coeffs = []
    for quant_sign, quant_name in zip(quant_signs, quant_names):
        krr_model = loadpkl(
            pkl_store_dir + "/models/krr_model_" + quant_name.replace(" ", "_") + ".pkl"
        )
        coeff = quant_sign / quant_QM9_stddev(quant_name)

        SLATM_model = SLATM_MMFF_based_model(krr_model)
        SLATM_model.verify_mbtypes(mmff_xyz_list)

        SLATM_models.append(SLATM_model)
        coeffs.append(coeff)

    min_functions.append(LinearCombination(SLATM_models, coeffs, quant_name))

for min_func, keyword in zip(min_functions, min_func_keywords):
    pkl_name = pkl_store_dir + "/minimized_function_" + keyword + ".pkl"
    dump2pkl(min_func, pkl_name)
