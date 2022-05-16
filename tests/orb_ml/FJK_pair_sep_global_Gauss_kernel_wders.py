from molopt.test_utils import dirs_xyz_list, logfile, timestamp
from molopt.orb_ml import OML_Slater_pair_list_from_xyzs
from molopt.orb_ml.kernels import gauss_sep_orb_sym_kernel, gauss_sep_orb_kernel, oml_ensemble_avs_stddevs
from molopt.orb_ml.representations import OML_rep_params
import numpy as np
import random, sys

seed=1
num_test_mols_1=50
num_test_mols_2=40

logfile_name=sys.argv[1]

test_xyz_dir="./qm7"
all_xyzs=dirs_xyz_list(test_xyz_dir)

tested_xyzs_1=random.Random(seed).sample(all_xyzs, num_test_mols_1)
tested_xyzs_2=random.Random(seed+1).sample(all_xyzs, num_test_mols_2)

logfile=logfile(logfile_name)

Slater_pair_kwargs={"second_oml_comp_kwargs" : {"used_orb_type" : "HOMO_removed", "calc_type" : "UHF"}}

oml_compounds_1=OML_Slater_pair_list_from_xyzs(tested_xyzs_1, **Slater_pair_kwargs)
oml_compounds_2=OML_Slater_pair_list_from_xyzs(tested_xyzs_2, **Slater_pair_kwargs)

orb_rep_params=OML_rep_params(max_angular_momentum=1, orb_atom_rho_comp=0.95)

timestamp("Representation generation.")
oml_compounds_1.generate_orb_reps(orb_rep_params, disable_openmp=True)
oml_compounds_2.generate_orb_reps(orb_rep_params, disable_openmp=False)
timestamp("Finished.")

timestamp("Standard deviation calculations.")
av_vals, width_params=oml_ensemble_avs_stddevs(oml_compounds_1)
timestamp("Finished.")

logfile.write("Width params")
logfile.write(width_params)

sigmas=np.array([0.5, *width_params])

logfile.write("xyz list 1")
logfile.write(tested_xyzs_1)

logfile.write("xyz list 2")
logfile.write(tested_xyzs_2)

logfile.write("kernel_11")
timestamp("Symmetrical kernel calculation.")
kernel_wders=gauss_sep_orb_sym_kernel(oml_compounds_1, sigmas, with_ders=True, global_Gauss=True)
timestamp("Finished.")
#logfile.export_matrix(kernel_wders)
logfile.randomized_export_3D_arr(kernel_wders, seed+2)

logfile.write("kernel_12")
timestamp("Asymmetrical kernel calculation.")
kernel_wders=gauss_sep_orb_kernel(oml_compounds_1, oml_compounds_2, sigmas, with_ders=True, global_Gauss=True)
timestamp("Finished.")
#logfile.export_matrix(kernel_wders)
logfile.randomized_export_3D_arr(kernel_wders, seed+2)
logfile.close()
