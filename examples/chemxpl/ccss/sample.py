import random
from bmapqml.chemxpl.minimized_functions import sample_local_space
import numpy as np
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
from bmapqml.utils import dump2tar, loadtar
from joblib import Parallel, delayed
from ccs_rdf import RDF_Plotter, mc_run
import pandas as pd

np.random.seed(1337)
random.seed(1337)


if __name__ == '__main__':

    bits = 2048
    nsim  = 48
    min_d, max_d, d_int = 0.5, 6.25, 0.25
    D_HIGH = np.arange(min_d, max_d, d_int)
    d_range    = (min_d, max_d, d_int)


    min_func_name, fp_type = "chemspacesampler", "MorganFingerprint"
    all_molecules = pd.read_csv("targets.csv")

    ALL_NAMES, ALL_SMILES = all_molecules["name"].values, all_molecules["smiles"].values

    
    for molecule, smiles in zip(ALL_NAMES, ALL_SMILES):
    # zip(["aspirin"], ["CC(=O)OC1=CC=CC=C1C(=O)O"]):
    #zip(ALL_NAMES, ALL_SMILES):
    #zip(["aspirin"], ["CC(=O)OC1=CC=CC=C1C(=O)O"]):
    #in  #zip(["aspirin"], ["CC(=O)OC1=CC=CC=C1C(=O)O"]):
    
        target = smiles
        output = xyz_list2mols_extgraph([f"./xyz/{molecule}.xyz"])
        X_init = rdkit_descriptors.get_all_FP([target],fp_type, nBits=bits) 
        print(target)
        RUN, PROCESS, PLOT = True, False, False #False, True, True
        save_path = f"/store/jan/trash/mc_sampling/rdf/{molecule}.pkl"
        respath = f"/store/jan/trash/mc_sampling/rdf"
        if RUN:
            for i1, dh in enumerate(D_HIGH): #problem at #i1 = 4,#dh = D_HIGH[i1] , and benzene
                dl = dh - d_int
                min_func = sample_local_space(X_init, verbose=False,check_ring=False, pot_type="flat_parabola",fp_type=fp_type , epsilon=5,gamma=dl, sigma=dh, nbits=bits)
                try:
                    Parallel(n_jobs=nsim)(delayed(mc_run)(output[0].chemgraph, min_func, f"{molecule}_{i1}_{i2}",respath) for i2 in range(240))
                except Exception as e:
                    print("Got following error:" , e)
                    pass #continue

        if PROCESS:
            
            n_shuffle  = 1
            rdf        = RDF_Plotter(d_range,target,molecule,n_shuffle,nBits=bits,ncpu=24,  verbose=True)
            N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN = rdf.sample_convergence_RDF()
            dump2tar([rdf.D_HIGH,N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN], save_path)

        if PLOT:
            RESULTS                                                                     = loadtar(save_path)
            D_HIGH,N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN  = RESULTS[0], RESULTS[1], RESULTS[2], RESULTS[3], RESULTS[4], RESULTS[5]
            rdf             = RDF_Plotter(d_range,target,molecule,nBits=bits,verbose=True)
            rdf.plot_rdf(N_SAMPLES,N_MEAN,N_EXT)
