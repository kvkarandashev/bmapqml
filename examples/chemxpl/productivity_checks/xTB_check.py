# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from bmapqml.orb_ml.tblite_interface import generate_pyscf_mf_mol
import os, random, sys
import numpy as np
from bmapqml.test_utils import timestamp


seed = 1
# Replace with path to QM9 directory
QM9_dir = os.environ["DATA"] + "/QM9_formatted"
test_num = 1000

xyz_list = dirs_xyz_list(QM9_dir)
random.seed(seed)
np.random.seed(seed)
random.shuffle(xyz_list)

timestamp("Reading xyzs")
test_comps = [
    OML_compound(xyz_file, software="tblite", calc_type="GFN2-xTB")
    for xyz_file in xyz_list[-test_num:]
]
timestamp("Done")

timestamp("Running xTB calculations with tblite")
sys.stdout = open(os.devnull, "w")
for test_comp in test_comps:
    _, _ = generate_pyscf_mf_mol(test_comp)
sys.stdout = sys.__stdout__
timestamp("Finished running the calculations")
