from bmapqml.test_utils import dirs_xyz_list, logfile, timestamp
from bmapqml.orb_ml import OML_Slater_pair_list_from_xyzs
from bmapqml.orb_ml.kernels import oml_ensemble_avs_stddevs
from bmapqml.orb_ml.representations import OML_rep_params
import numpy as np
import random, sys

# Toggle to use Fortran or Numba implementation.
if len(sys.argv) < 3:
    use_Fortran = True
else:
    use_Fortran = sys.argv[2] == "True"

seed = 1
num_test_mols_1 = 50
num_test_mols_2 = 40

logfile_name = sys.argv[1]

test_xyz_dir = "./qm7"
all_xyzs = dirs_xyz_list(test_xyz_dir)

tested_xyzs_1 = random.Random(seed).sample(all_xyzs, num_test_mols_1)
tested_xyzs_2 = random.Random(seed + 1).sample(all_xyzs, num_test_mols_2)

logfile = logfile(logfile_name)

Slater_pair_kwargs = {
    "second_oml_comp_kwargs": {"used_orb_type": "HOMO_removed", "calc_type": "UHF"}
}

oml_compounds_1 = OML_Slater_pair_list_from_xyzs(tested_xyzs_1, **Slater_pair_kwargs)
oml_compounds_2 = OML_Slater_pair_list_from_xyzs(tested_xyzs_2, **Slater_pair_kwargs)

orb_rep_params = OML_rep_params(max_angular_momentum=1, orb_atom_rho_comp=0.95)

rep_calc = timestamp()  # "Representation generation start:")
oml_compounds_1.generate_orb_reps(orb_rep_params, fixed_num_threads=1)
oml_compounds_2.generate_orb_reps(orb_rep_params)
timestamp("Representation generation duration:", rep_calc)

sd_calc = timestamp()  # "Standard deviation calculations start:")
av_vals, width_params = oml_ensemble_avs_stddevs(oml_compounds_1)
timestamp("Standard deviation calculations duration:", sd_calc)

logfile.write("Width params")
logfile.write(width_params)

sigmas = np.array([0.5, *width_params])

logfile.write("xyz list 1")
logfile.write(tested_xyzs_1)

logfile.write("xyz list 2")
logfile.write(tested_xyzs_2)

if use_Fortran:
    from bmapqml.orb_ml.fkernels import gauss_sep_orb_sym_kernel, gauss_sep_orb_kernel
else:
    from bmapqml.orb_ml.kernels import gauss_sep_orb_sym_kernel, gauss_sep_orb_kernel

logfile.write("kernel_11")
sym_kern_calc = timestamp()  # "Symmetrical kernel calculation start:")
kernel_wders = gauss_sep_orb_sym_kernel(
    oml_compounds_1, sigmas, with_ders=True, global_Gauss=True
)
timestamp("Symmetrical kernel calculation duration:", sym_kern_calc)
# logfile.export_matrix(kernel_wders)
logfile.randomized_export_3D_arr(kernel_wders, seed + 2)

logfile.write("kernel_12")
asym_kern_calc = timestamp()  # "Asymmetrical kernel calculation start:")
kernel_wders = gauss_sep_orb_kernel(
    oml_compounds_1, oml_compounds_2, sigmas, with_ders=True, global_Gauss=True
)
timestamp("Asymmetrical kernel calculation duration:", asym_kern_calc)
# logfile.export_matrix(kernel_wders)
logfile.randomized_export_3D_arr(kernel_wders, seed + 2)
logfile.close()
