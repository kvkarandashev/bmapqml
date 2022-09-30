from bmapqml.chemxpl.minimized_functions import  local_space_sampling
import numpy as np
from ccs_rdf import mc_run
from ccs_rdf import loadXYZ
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
from qml.utils.alchemy import *
molecule="benzene"
chgs, crds, _ = loadXYZ("benzene.xyz")
chgs = np.array([NUCLEAR_CHARGE[q] for q in chgs])

output = xyz_list2mols_extgraph([f"benzene.xyz"])
X_init = local_space_sampling.gen_soap(crds,chgs) # gen_fchl(crds, chgs) #gen_bob(crds, chgs)

dl, dh = 450, 500
respath = "."

min_func = local_space_sampling.sample_local_space_3d(X_init,chgs, verbose=False,epsilon=50,gamma=dl, sigma=dh, repfct=local_space_sampling.gen_soap)
mc_run(output[0].chemgraph, min_func, f"{molecule}",respath, num_MC_steps=10)
