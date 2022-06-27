from bmapqml.dataset_processing.qm9_format_specs import read_InChI
from bmapqml.chemxpl.utils import InChI_to_egc, xyz2mol_extgraph, RdKitFailure
import numpy as np
#from bmapqml.utils import safe_array_func_eval
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
import sys, glob

if len(sys.argv) < 3:
    print("Use QM9 xyz directory as first argument and bad mol id output as second.")
    quit()

QM9_dir=sys.argv[1]
output_file=sys.argv[2]

def xyz_InChI_consistent(xyz_file):
    InChI=read_InChI(xyz_file)
    try:
        egc1=InChI_to_egc(InChI)
    except InvalidAdjMat:
        return False
    except RdKitFailure:
        return False
    try:
        egc2=xyz2mol_extgraph(xyz_file)
    except:
        return False
    return egc1==egc2

xyz_files=glob.glob(QM9_dir+"/*.xyz")
xyz_files.sort()

consistency=[]
for xyz in xyz_files:
    print("CHECKING:", xyz)
    consistency.append(xyz_InChI_consistent(xyz))

print("FINISHED")

f=open(output_file, 'w')
for i, cons in enumerate(consistency):
    if not cons:
        print(i, file=f)
f.close()
