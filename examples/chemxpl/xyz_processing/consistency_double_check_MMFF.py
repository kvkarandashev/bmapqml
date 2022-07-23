# Double-check that all generated MMFF coordinates have the same topology as the starting points from QM9 directory.
# (Created for testing purposes)
import os
from bmapqml.dataset_processing.qm9_format_specs import read_SMILES
from bmapqml.chemxpl.utils import xyz2mol_extgraph, SMILES_to_egc
from bmapqml.test_utils import dirs_xyz_list

MMFF_xyzs_dir = os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs"
QM9_xyzs_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"

for i, mmff_xyz_file in enumerate(dirs_xyz_list(MMFF_xyzs_dir)):
    print("Checking:", i, mmff_xyz_file)
    true_xyz_name = os.path.basename(mmff_xyz_file)
    qm9_xyz_name = QM9_xyzs_dir + "/" + true_xyz_name
    SMILES = read_SMILES(qm9_xyz_name)
    egc_SMILES = SMILES_to_egc(SMILES)
    egc_coords = xyz2mol_extgraph(qm9_xyz_name)
    egc_mmff_coords = xyz2mol_extgraph(mmff_xyz_file)
    if (
        (egc_SMILES != egc_coords)
        or (egc_SMILES != egc_mmff_coords)
        or (egc_mmff_coords != egc_coords)
    ):
        print(
            mmff_xyz_file,
            (egc_SMILES != egc_coords),
            (egc_SMILES != egc_mmff_coords),
            (egc_mmff_coords != egc_coords),
        )
        print("SMILES:", egc_SMILES)
        print("From coords:", egc_coords)
        print("From MMFF coords:", egc_mmff_coords)
        quit()
print("All checked")
