from bmapqml.dataset_processing.electrolyte_genome_format_specs import Quantity
import os

EGP_dir = os.environ["DATA"] + "/EGP/new_xyzs"

example_xyzs = ["mol-50089.xyz", "mol-64060.xyz"]

for quant_name in ["IE", "EA"]:
    print(quant_name)
    quantity = Quantity(quant_name)
    for xyz in example_xyzs:
        true_xyz = EGP_dir + "/" + xyz
        true_quant = quantity.extract_xyz(true_xyz)
        est_quant = quantity.OML_calc_quant(true_xyz, basis="cc-pvdz", calc_type="UHF")
        print(true_quant, est_quant)
