# An example of using stochatic gradient descent to optimize sigma value for Coulomb Matrix with Gaussian kernel.
from bmapqml.krr import KRR, learning_curve
from bmapqml.dataset_processing.qm9_format_specs import Quantity
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.orb_ml import OML_compound
from qml.representations import get_slatm_mbtypes, generate_slatm
import os, random
import numpy as np

# Replace with path to QM9 directory. 
QM9_dir=os.environ["DATA"]+"/QM9_filtered/xyzs"

quant_name='HOMO-LUMO gap'
seed=1

# get_quants_comps

# Replace with path to QM9 directory
MMFF_QM9_dir=os.environ["DATA"]+"/QM9_filtered/MMFF_xyzs"

train_nums=[1000, 2000, 4000, 8000]
bigger_train_subset=32000
hyperparam_opt_num=2000
check_num=8000

size=29 # maximum number of atoms in a QM9 molecule.

mmff_xyz_list=dirs_xyz_list(MMFF_QM9_dir)
random.seed(seed)
random.shuffle(mmff_xyz_list)

def get_quants_comps(mmff_xyz_list, quantity):
    quant_vals=[]
    for mmff_xyz_file in mmff_xyz_list:
        true_xyz_file=QM9_dir+"/"+os.path.basename(mmff_xyz_file)
        quant_vals.append(quantity.extract_xyz(true_xyz_file))

    quant_vals=np.array(quant_vals)
    comps=[OML_compound(mmff_xyz_file) for mmff_xyz_file in mmff_xyz_list]

    ncharges=np.zeros((len(mmff_xyz_list), size), dtype=int)
    comps=[]

    for i, mmff_xyz_file in enumerate(mmff_xyz_list):
        comp=OML_compound(mmff_xyz_file)
        ncharges[i, :len(comp.nuclear_charges)]=comp.nuclear_charges[:]
        comps.append(comp)

    mbtypes=get_slatm_mbtypes(ncharges)

    representations=np.array([generate_slatm(comp.coordinates, comp.nuclear_charges, mbtypes) for comp in comps])
    return representations, comps, np.array(quant_vals)

quant=Quantity(quant_name)
all_training_reps, all_training_comps, all_training_quants=get_quants_comps(mmff_xyz_list[:bigger_train_subset], quant)

KRR_model=KRR()

KRR_model.optimize_hyperparameters(all_training_reps[:hyperparam_opt_num], all_training_quants[:hyperparam_opt_num])

check_reps, check_comps, check_quants=get_quants_comps(mmff_xyz_list[-check_num:], quant)

MAEs=learning_curve(KRR_model, all_training_reps, all_training_quants, check_reps, check_quants, train_nums)

print("Quantity: ", quant_name, ", Ntrain and MAEs:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, MAE_line)

print("Compact form of the lc:")
for train_num, MAE_line in zip(train_nums, MAEs):
    print(train_num, np.mean(MAE_line), np.std(MAE_line))
