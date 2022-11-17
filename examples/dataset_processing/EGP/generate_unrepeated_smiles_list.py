# Generate a text file containing all unrepeated SMILES of molecules encountered in EGP xyzs.
from bmapqml.chemxpl.utils import SMILES_to_egc, InvalidAdjMat, RdKitFailure
from bmapqml.chemxpl.test_utils import egc_list_nhatoms_hist, elements_in_egc_list
from bmapqml.dataset_processing.electrolyte_genome_format_specs import Quantity
from bmapqml.test_utils import dirs_xyz_list
import os
import numpy as np
from sortedcontainers import SortedList

EGP_dir = os.environ["DATA"] + "/EGP/new_xyzs"

SMILES_func = Quantity("smiles")

charge_func = Quantity("charge")

xyz_list = dirs_xyz_list(EGP_dir)

output_egcs = SortedList()

err_log = open("egc_check_err.log", "w")

for xyz in xyz_list:
    charge = charge_func.extract_xyz(xyz)
    merged_SMILES = SMILES_func.extract_xyz(xyz)
    if charge != 0:
        print("### Not neutral:", merged_SMILES, xyz, file=err_log)
        continue
    smiles_list = merged_SMILES.split(".")
    for smiles in smiles_list:
        try:
            egc = SMILES_to_egc(smiles)
        except InvalidAdjMat:
            print("///Non-translatable SMILES:", smiles, xyz, file=err_log)
            continue
        except RdKitFailure:
            print("///RdKit failure:", smiles, xyz, file=err_log)
            continue
        if np.all(egc.nuclear_charges == 1):
            print("///H-only molecule found in:", xyz, file=err_log)
            continue
        if egc not in output_egcs:
            output_egcs.add(egc)

err_log.close()

egc_output = open("chemgraph_str.txt", "w")
SMILES_output = open("SMILES.txt", "w")

for egc in output_egcs:
    print(egc, file=egc_output)
    print(egc.additional_data["SMILES"], file=SMILES_output)

egc_output.close()
SMILES_output.close()

size_hist = egc_list_nhatoms_hist(output_egcs)

summary_output = open("data_summary.txt", "w")
print("Elements present:", elements_in_egc_list(output_egcs), file=summary_output)
for size, num in size_hist.items():
    print(size, num, file=summary_output)
summary_output.close()
