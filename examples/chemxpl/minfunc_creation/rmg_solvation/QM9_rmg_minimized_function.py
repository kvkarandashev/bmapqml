from bmapqml.chemxpl.minimized_functions.rmgpy_quantity_estimates import RMGSolvation
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
from bmapqml.chemxpl.minimized_functions.quantity_estimates import (
    ConstrainedQuant,
    LinearCombination,
)
from bmapqml.chemxpl.utils import xyz2mol_extgraph, InvalidAdjMat
from bmapqml.utils import dump2pkl, embarrassingly_parallel
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.test_utils import dirs_xyz_list
import sys
import numpy as np

QM9_dir = "/data/konst/QM9_formatted"

xyzs = dirs_xyz_list(QM9_dir)

if len(sys.argv) == 1:
    num_attempts = 0
else:
    num_attempts = int(sys.argv[1])

if len(sys.argv) == 3:
    NPROCS = int(sys.argv[2])
else:
    NPROCS = 1

forcefield = "MMFF"

solvent = "water"

true_func = RMGSolvation(
    num_attempts=num_attempts, ff_type=forcefield, solvent_label=solvent
)

restricted_ncharges = []  # [8, 9]

constraints = [NoProtonation(restricted_ncharges=restricted_ncharges)]

constr_func = ConstrainedQuant(true_func, "RMGSolvation", constraints=constraints)

cur_id = 0


def xyz_val(xyz):
    try:
        egc = xyz2mol_extgraph(xyz)
    except InvalidAdjMat:
        return None
    tp = TrajectoryPoint(egc=egc)
    return constr_func(tp)


all_vals = embarrassingly_parallel(xyz_val, xyzs, (), num_procs=NPROCS)

vals = []
for cur_val in all_vals:
    if cur_val is not None:
        vals.append(cur_val)

std = np.std(vals)

print("Number of values:", len(vals))
print("Mean:", np.mean(vals))
print("Standard deviation:", std)

finalized_quant = LinearCombination([constr_func], [1.0 / std], "ConstrainedSolvation")
dump2pkl(finalized_quant, "RMGNormalizedSolvation_" + str(num_attempts) + ".pkl")
