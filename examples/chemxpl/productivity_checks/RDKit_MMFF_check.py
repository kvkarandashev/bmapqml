# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from bmapqml.orb_ml.tblite_interface import generate_pyscf_mf_mol
import os, random, sys
import numpy as np
from bmapqml.test_utils import timestamp
from bmapqml.chemxpl.utils import SMILES_to_egc, egc_with_coords, MMFFInconsistent
from bmapqml.dataset_processing.qm9_format_specs import read_SMILES

seed = 1
# Replace with path to QM9 directory
MMFF_QM9_dir = os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs"
QM9_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"
test_num = 1000
num_mmff_attempts = 10

xyz_list = dirs_xyz_list(MMFF_QM9_dir)
random.seed(seed)
np.random.seed(seed)
random.shuffle(xyz_list)

true_xyz_list = [
    QM9_dir + "/" + os.path.basename(xyz_file) for xyz_file in xyz_list[-test_num:]
]
timestamp("Reading SMILES")
test_SMILES = [read_SMILES(xyz_file) for xyz_file in true_xyz_list]
timestamp("Finished reading")

timestamp("Making egcs")
test_egcs = [SMILES_to_egc(SMILES) for SMILES in test_SMILES]
timestamp("Finished")

timestamp("Making coordinates")
for SMILES, test_egc, test_xyz in zip(test_SMILES, test_egcs, true_xyz_list):
    #    print(SMILES, test_egc, test_xyz)
    for _ in range(num_mmff_attempts):
        try:
            test_egc_wcoords = egc_with_coords(test_egc)
        except MMFFInconsistent:
            #            print("REDOING")
            continue
        break
timestamp("Finished")
