import glob, sys, subprocess
from bmapqml.dataset_processing.qm9_format_specs import xyz_SMILES_consistent
from bmapqml.utils import mkdir

if len(sys.argv)==1:
    print("Use location of QM9 xyzs as the script's argument.")
    quit()
else:
    QM9_dir=sys.argv[1]

xyz_list=glob.glob(QM9_dir+"/*.xyz")

new_dirname="QM9_filtered"
mkdir(new_dirname)

for xyz in xyz_list:
    if xyz_SMILES_consistent(xyz):
        subprocess.run(["cp", xyz, new_dirname])

