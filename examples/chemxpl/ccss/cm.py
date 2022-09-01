import random
from bmapqml.chemxpl.minimized_functions import sample_local_space_3d
import numpy as np
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
from bmapqml.utils import dump2tar, loadtar
from joblib import Parallel, delayed
from ccs_rdf import RDF_Plotter, mc_run
import pandas as pd
import pdb
import qml
from qml.representations import *
np.random.seed(1337)
random.seed(1337)



def loadXYZ(filename, ang2bohr=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
        numAtoms = int(lines[0])
        positions = np.zeros((numAtoms, 3), dtype=np.double)
        elems = [None] * numAtoms
        comment = lines[1]
        for x in range (2, 2 + numAtoms):
            line_split = lines[x].rsplit()
            elems[x - 2] = line_split[0]
            
            line_split[1] = line_split[1].replace('*^', 'E')
            line_split[2] = line_split[2].replace('*^', 'E')
            line_split[3] = line_split[3].replace('*^', 'E')
            
            positions[x - 2][0] = np.double(line_split[1]) 
            positions[x - 2][1] = np.double(line_split[2]) 
            positions[x - 2][2] = np.double(line_split[3])
                
    return np.asarray(elems), np.asarray(positions), comment



if __name__ == '__main__':


    _, crds, _ = loadXYZ("/home/jan/projects/MOLOPT/do_sim/rdf/xyz/benzene.xyz")
    #pdb.set_trace()
    chgs = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1 ]
    
    cm = generate_coulomb_matrix(chgs, crds,
                                size=50, sorting="row-norm")
    
    min_func_name, fp_type = "chemspacesampler", "MorganFingerprint"
    output = xyz_list2mols_extgraph([f"/home/jan/projects/MOLOPT/do_sim/rdf/xyz/benzene.xyz"])
    X_init = generate_coulomb_matrix(chgs, crds,
                            size=50, sorting="row-norm")
    save_path = f"/store/jan/trash/test_stuff.pkl"
    respath = f"/store/jan/trash"

    min_func = sample_local_space_3d(X_init, verbose=True,epsilon=5,gamma=0.6, sigma=0.9)
    mc_run(output[0].chemgraph, min_func, f"test_stuff",respath)