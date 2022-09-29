import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import random
from bmapqml.chemxpl.minimized_functions import  local_space_sampling
import numpy as np
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
from bmapqml.utils import dump2tar, loadtar
from joblib import Parallel, delayed
from ccs_rdf import RDF_Plotter, mc_run, loadXYZ
import pandas as pd
import pdb
import qml
from qml.representations import *
from qml.utils.alchemy import *

np.random.seed(1337)
random.seed(1337)


if __name__ == '__main__':

    molecule="benzene"
    chgs, crds, _ = loadXYZ("./xyz/benzene.xyz")
    chgs = np.array([NUCLEAR_CHARGE[q] for q in chgs])

    output = xyz_list2mols_extgraph([f"./xyz/benzene.xyz"])
    X_init = local_space_sampling.gen_soap(crds,chgs) # gen_fchl(crds, chgs) #gen_bob(crds, chgs)

    save_path = f"/store/jan/trash/mc_samplting/rdf_3d/test_stuff.pkl"
    respath = f"/store/jan/trash/mc_sampling/rdf_3d"

    min_d, max_d, d_int = 50, 500, 50
    D_HIGH = np.arange(min_d, max_d, d_int)
    d_range    = (min_d, max_d, d_int)
    target = "C1=CC=CC=C1"


    RUN, PROCESS, PLOT = True, False, False

    if RUN:
        for i1, dh in enumerate(D_HIGH):
        #dl, dh  = 150, 155 #60, 65 FOR CM AND BOB
        #dl, dh  = 0, 90 #120, 125 
        # 50, 60 #60, 65 FOR FCHL, SOAP
            dl = dh - d_int
            min_func = local_space_sampling.sample_local_space_3d(X_init,chgs, verbose=False,epsilon=50,gamma=dl, sigma=dh, repfct=gen_soap)
            #mc_run(output[0].chemgraph, min_func, f"{molecule}_{i1}_0",respath)
            Parallel(n_jobs=1)(delayed(mc_run)(output[0].chemgraph, min_func, f"{molecule}_{i1}_{i2}",respath, num_MC_steps=10000 ) for i2 in range(240))
            
    if PROCESS:
        n_shuffle  = 1
        rdf        = RDF_Plotter(respath,d_range,target,molecule,n_shuffle,nBits=2048,ncpu=24,  verbose=True)
        N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN = rdf.sample_convergence_RDF()
        #dump2tar([rdf.D_HIGH,N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN], save_path)
