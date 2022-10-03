from bmapqml.chemxpl.minimized_functions.rmgpy_quantity_estimates import RMGSolvation
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
from bmapqml.chemxpl.minimized_functions.quantity_estimates import (
    ConstrainedQuant,
    LinearCombination,
)
from bmapqml.chemxpl.utils import xyz2mol_extgraph, InvalidAdjMat
from bmapqml.utils import dump2pkl  # , embarrassingly_parallel
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.test_utils import dirs_xyz_list
import sys
import numpy as np

QM9_dir = "/data/konst/QM9_formatted"

xyzs = sorted(dirs_xyz_list(QM9_dir))

if len(sys.argv) == 1:
    num_attempts = 0
else:
    num_attempts = int(sys.argv[1])

forcefield = "MMFF"

solvent = "water"

true_func = RMGSolvation(
    num_attempts=num_attempts, ff_type=forcefield, solvent_label=solvent
)

constraints = [NoProtonation(restricted_ncharges=[8, 9])]

constr_func = ConstrainedQuant(true_func, "RMGSolvation", constraints=constraints)

vals = []

cur_id = 0

for xyz in xyzs:
    try:
        egc = xyz2mol_extgraph(xyz)
    except InvalidAdjMat:
        continue
    tp = TrajectoryPoint(egc=egc)
    cur_val = constr_func(tp)
    if cur_val is not None:
        cur_id += 1
        vals.append(cur_val)
        print(cur_id, egc.chemgraph.nhatoms(), egc)

std = np.std(vals)

print("Number of values:", len(vals))
print("Mean:", np.mean(vals))
print("Standard deviation:", std)

finalized_quant = LinearCombination([constr_func], [1.0 / std], "ConstrainedSolvation")
dump2pkl(finalized_quant, "NormalizedSolvation_" + str(num_attempts) + ".pkl")
