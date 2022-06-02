# Imports xyz files created with mmff_coord_estimates.py as ExtGraphCompound objects.
import glob
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph

xyz_names=glob.glob("./*.xyz")

egc_list=xyz_list2mols_extgraph(xyz_names, xyz_to_add_data=True)

for egc in egc_list:
    print(egc, egc.additional_data["xyz"])
