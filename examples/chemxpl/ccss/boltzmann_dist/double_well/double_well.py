import random
from bmapqml.chemxpl.minimized_functions import sample_local_space
import numpy as np
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
from joblib import Parallel, delayed
from ccs_rdf import mc_run
import pandas as pd
from bmapqml.chemxpl.plotting import Analyze
import pdb
import matplotlib.pyplot as plt

np.random.seed(1337)
random.seed(1337)

def double_well_potential(d, epsilon):
    """
    Double well potential with barrier height epsilon 
    """
    fc = 0.3
    return epsilon*((fc*d-1)**4 - 2*(fc*d-1)**2 + 1)


if __name__ == '__main__':

    bits = 2048
    nsim  = 2


    
    min_func_name, fp_type = "chemspacesampler", "MorganFingerprint"
    molecule, smiles = "oxazine", "C1=CNOC=C1"

    target_center, target_init = smiles, "C1=CONO1"
    output_center = xyz_list2mols_extgraph([f"./{molecule}.xyz"])
    X_center = rdkit_descriptors.get_all_FP([target_center],fp_type, nBits=bits) 
    output_init = xyz_list2mols_extgraph([f"./init.xyz"])
    X_init = rdkit_descriptors.get_all_FP([target_init],fp_type, nBits=bits)

    RUN, PROCESS = True, True
    save_path = f"/store/jan/trash/mc_sampling/double_well/{molecule}.pkl"
    respath = f"/store/jan/trash/mc_sampling/double_well"

    num_MC_steps = 5000
    ref_beta = 10
    epsilon = 100

    if RUN:
        betas=[None,ref_beta, ref_beta/5]
        min_func = sample_local_space(X_center, verbose=True, pot_type="double_well",fp_type=fp_type , epsilon=epsilon,gamma=1, sigma=1, nbits=bits)
        try:
            Parallel(n_jobs=2)(delayed(mc_run)(output_init[0].chemgraph, min_func, f"{molecule}_{i1}",respath,betas,num_MC_steps,bias_coeff=None,vbeta_bias_coeff=None,bound_enforcing_coeff=None ) for i1 in range(0,2))
        except:
            print("Error")
            pass

    if PROCESS:

        ana   = Analyze(f"{respath}"+f"/{molecule}_*.pkl",full_traj=True, verbose=True, mode="sampling")
        ALL_HISTOGRAMS,GLOBAL_HISTOGRAM, ALL_TRAJECTORIES = ana.parse_results()        
        
        #pdb.set_trace()
        #study how often for a given number of steps molecules
        #from d > barrier appear in a simulation at low temperature
        #due to replica exchange compared to no replica exchange


        for sample in ALL_TRAJECTORIES:
            for ind_b, b in enumerate(sample):
                SMILES = list(b["SMILES"].values.flatten()) 
                #pdb.set_trace()
                darr = np.zeros(len(SMILES))
                for i, s in  enumerate(SMILES):
                    darr[i] = np.linalg.norm(X_center - ana.compute_representations([s], nBits=bits))

                #indices of darr where d > 3.3
                idx = len(np.where(darr < 1.5)[0])
                #print number of molecules in the intervall
                print(f"{ind_b} Number of molecules with d < 1.5: {idx}")
        exit()


        #flatten the list of ALL_SMILES
        ALL_SMILES = [item for sublist in ALL_SMILES for item in sublist]
        #pdb.set_trace()
        darr = np.zeros(len(ALL_SMILES))
        for i, s in  enumerate(ALL_SMILES):
            darr[i] = np.linalg.norm(X_init - ana.compute_representations([s], nBits=bits))
