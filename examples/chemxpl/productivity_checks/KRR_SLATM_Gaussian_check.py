# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from qml.representations import get_slatm_mbtypes, generate_slatm
from bmapqml.fkernels import gaussian_kernel_matrix
import os, random
import numpy as np
from bmapqml.test_utils import timestamp

seed = 1
# Replace with path to QM9 directory
QM9_dir = os.environ["DATA"] + "/QM9_formatted"
train_num = 100000
test_num = 1000

xyz_list = dirs_xyz_list(QM9_dir)
random.seed(seed)
np.random.seed(seed)
random.shuffle(xyz_list)


def get_reps(xyz_list, mbtypes=None):
    comps = [OML_compound(xyz_file) for xyz_file in xyz_list]
    if mbtypes is None:
        timestamp("Calculating mbtypes")
        maxsize = max(len(comp.nuclear_charges) for comp in comps)
        all_nuclear_charges = np.zeros((len(comps), maxsize), dtype=int)
        for cid, comp in enumerate(comps):
            all_nuclear_charges[
                cid, : len(comp.nuclear_charges)
            ] = comp.nuclear_charges[:]

        mbtypes = get_slatm_mbtypes(all_nuclear_charges)
        timestamp("Done")
    timestamp("Calculating representations")
    representations = np.array(
        [
            generate_slatm(comp.coordinates, comp.nuclear_charges, mbtypes)
            for comp in comps
        ]
    )
    timestamp("Done")
    return representations, mbtypes


training_reps, mbtypes = get_reps(xyz_list[:train_num])

KRR_model = KRR(kernel_function=gaussian_kernel_matrix)

KRR_model.X_train = training_reps
KRR_model.alphas = np.random.random((len(KRR_model.X_train),))
timestamp("Initial sigma guess")
KRR_model.sigmas = KRR_model.opt_hyperparameter_guess(training_reps[:2000], None)[1:2]
timestamp("Finished calculating initial sigma guess")

test_reps, _ = get_reps(xyz_list[-test_num:], mbtypes=mbtypes)

timestamp("Making predictions without optimizing the number of python calls:")

test_preds = [
    KRR_model.predict(test_reps[i : i + 1])[0] for i in range(X_test.shape[0])
]

timestamp("Ended making predictions")
