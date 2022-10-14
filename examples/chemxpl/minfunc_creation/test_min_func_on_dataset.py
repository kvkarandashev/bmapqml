from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.utils import xyz2mol_extgraph, trajectory_point_to_canonical_rdkit
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
import os, sys
from bmapqml.utils import dump2pkl, embarrassingly_parallel, loadpkl

quant_pkl = sys.argv[1]

NPROCS = int(sys.argv[2])

quant = loadpkl(quant_pkl)

val_str = "val"
tp_str = "tp"


def xyz_data_gen(xyz_name):
    output = {"xyz": xyz_name}
    try:
        tp = TrajectoryPoint(egc=xyz2mol_extgraph(xyz_name))
    except InvalidAdjMat:
        print(xyz_name, None)
        return output
    quant_val = quant(tp)
    print(xyz_name, quant_val)
    return {val_str: quant_val, tp_str: tp, **output}


def all_xyzs_data_extract(xyz_list):
    return embarrassingly_parallel(xyz_data_gen, xyz_list, (), num_procs=NPROCS)


xyzs = sorted(dirs_xyz_list(sys.argv[3]))

if len(sys.argv) < 5:
    dump_file = None
else:
    dump_file = sys.argv[4]

if (dump_file is not None) and os.path.isfile(dump_file):
    xyz_data = loadpkl(dump_file)
else:
    xyz_data = all_xyzs_data_extract(xyzs)
    if dump_file is not None:
        dump2pkl(xyz_data, dump_file)

min_val = None
min_tp = None

for entry in xyz_data:
    if val_str in entry:
        val = entry[val_str]
        if val is not None:
            if (min_val is None) or (min_val > val):
                min_val = val
                min_tp = entry[tp_str]

print("Minimal value:", min_val)
print("Minimal tp:", min_tp)
print(
    "Minimal tp SMILES:", trajectory_point_to_canonical_rdkit(min_tp, SMILES_only=True)
)
