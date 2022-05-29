import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import numpy as np
import os
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem  
import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from qml.utils import alchemy
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import DataStructs
import collections
import pdb
import deepchem as dc

random.seed(1337)
np.random.seed(1337)

"""
=========================================================================================================
  Ele-    ZPVE         U (0 K)      U (298.15 K)    H (298.15 K)    G (298.15 K)     CV
  ment   Hartree       Hartree        Hartree         Hartree         Hartree        Cal/(Mol Kelvin)
=========================================================================================================
   H     0.000000     -0.500273      -0.498857       -0.497912       -0.510927       2.981
   C     0.000000    -37.846772     -37.845355      -37.844411      -37.861317       2.981
   N     0.000000    -54.583861     -54.582445      -54.581501      -54.598897       2.981
   O     0.000000    -75.064579     -75.063163      -75.062219      -75.079532       2.981
   F     0.000000    -99.718730     -99.717314      -99.716370      -99.733544       2.981
=========================================================================================================
"""


def ExplicitBitVect_to_NumpyArray(fp_vec):

    fp2 = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp_vec, fp2)
    return fp2

    #return np.array(list(intmap))



def get_single_FP(smi, fp_type):
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smi)

    if fp_type=="MorganFingerprint":
        fp_mol = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=3,
            nBits=4096,
            useFeatures=True,
        )

    if fp_type=="TopologicalTorsion":
        fp_mol = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,  nBits=4096)
    if fp_type=="Avalon":
        fp_mol = GetAvalonFP(mol, nBits=4096)

    return fp_mol

def get_all_FP(SMILES, fp_type):

    X = []
    for smi in tqdm(SMILES):
        x = ExplicitBitVect_to_NumpyArray(get_single_FP(smi, fp_type))
        X.append(x)
    X = np.array(X)

    return X


def atomization_en(EN, ATOMS, normalize=False):
    
    en_H = -0.500273 
    en_C = -37.846772
    en_N = -54.583861
    en_O = -75.064579 
    en_F = -99.718730
    COMP =  collections.Counter(ATOMS)

    #ATOMIZATION = EN - (COMP['H']*en_H + COMP['C']*en_C + COMP['N']*en_N +  COMP['O']*en_O +  COMP['F']*en_F)
    #N^tot = Number of H-atoms x 1 + Number of C-atoms x 4 + Number of N-atoms x 3 + Number of O-atoms x 2 + Number of F-atoms x1
    #you divide atomization energy by N^tot and you're good
    if normalize:
        Ntot = (COMP['C']*4 + COMP['N']*3 +  COMP['O']*2 +  COMP['F']*1+COMP['H']*1)
        ATOMIZATION = (EN - (COMP['H']*en_H + COMP['C']*en_C + COMP['N']*en_N +  COMP['O']*en_O +  COMP['F']*en_F))
        return ATOMIZATION/Ntot
    
    else:
        ATOMIZATION = (EN - (COMP['H']*en_H + COMP['C']*en_C + COMP['N']*en_N + COMP['O']*en_O + COMP['F']*en_F))
        return ATOMIZATION
  
def read_xyz(path):
    """
    Reads the xyz files in the directory on 'path'
    Input
    path: the path to the folder to be read
    
    Output
    atoms: list with the characters representing the atoms of a molecule
    coordinates: list with the cartesian coordinates of each atom
    smile: list with the SMILE representation of a molecule
    prop: list with the scalar properties
    """
    atoms = []
    coordinates = []

    with open(path, 'r') as file:
        lines = file.readlines()
        n_atoms = int(lines[0])  # the number of atoms
        smile = lines[n_atoms + 3].split()[0]  # smiles string
        prop = lines[1].split()[2:]  # scalar properties
        
        # to retrieve each atmos and its cartesian coordenates
        for atom in lines[2:n_atoms + 2]:
            line = atom.split()
            # which atom
            atoms.append(line[0])

            # its coordinate
            # Some properties have '*^' indicading exponentiation 
            try:
                coordinates.append(
                    (float(line[1]),
                     float(line[2]),
                     float(line[3]))
                    )
            except:
                coordinates.append(
                    (float(line[1].replace('*^', 'e')),
                     float(line[2].replace('*^', 'e')),
                     float(line[3].replace('*^', 'e')))
                    )
                    
    return atoms, coordinates, smile, prop

def gen_conf(SMILES):

    RDKITS = []
    FAILED = []

    for ind, smi in enumerate(SMILES):

        try:
            mol=Chem.MolFromSmiles(smi)
            mol=Chem.AddHs(mol, explicitOnly=True)
            AllChem.EmbedMolecule(mol)
            #AllChem.MMFFOptimizeMolecule(mol)
            AllChem.UFFOptimizeMoleculeConfs(mol)
            RDKITS.append(mol)

        except Exception as e:
            FAILED.append(ind)
    
    RDKITS,FAILED = np.array(RDKITS), np.array(FAILED)
    return RDKITS,FAILED
    

def process_qm9(directory):
    len(os.listdir(directory))

    file = os.listdir(directory)[0]
    with open(directory+file, 'r') as f:
        content = f.readlines()


    data = []
    smiles = []
    properties = []
    for file in tqdm(os.listdir(directory)[:133884]):
        path = os.path.join(directory, file)
        atoms, coordinates, smile, prop = read_xyz(path)
        # A tuple with the atoms and its coordinates
        data.append((atoms, coordinates))
        smiles.append(smile)  # The SMILES representation

        ATOMIZATION = atomization_en(float(prop[10]), atoms, normalize=False)
        prop += [ATOMIZATION]
        properties.append(prop)  # The molecules properties

    properties_names = ['A', 'B', 'C', 'mu', 'alfa', 'homo', 'lumo', 'gap', 'R²', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'atomization']
    df = pd.DataFrame(properties, columns = properties_names) #.astype('float32')
    df['smiles'] = smiles
    df.head()

    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['mol'].isnull().sum()


    def canonize(mol):
        return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

    canon_smile = []
    for molecule in smiles:
        canon_smile.append(canonize(molecule))
        
    df['canon_smiles'] = canon_smile
    df['canon_smiles'][df['canon_smiles'].duplicated()]

    ind = df.index[df['canon_smiles'].duplicated()]
    df = df.drop(ind)
    df['mol'] = df['canon_smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df.to_csv('qm9.csv', index=False)
    return df


if __name__ == "__main__":
    process=True
    TARGET_PROPERTY = 'atomization'

    if process:
        df = process_qm9('/store/jan/datasets/qm9/')
    else:
        df = pd.read_csv('qm9.csv')


    

    SMILES, y  = df['canon_smiles'].values, np.float_(df[TARGET_PROPERTY].values)
    inds = np.arange(len(SMILES))
    np.random.shuffle(inds)
    SMILES, y = SMILES[inds], y[inds]


    param_grid = [{"krr__gamma": np.logspace(-11, -6, num=50), "krr__alpha": [1e-8, 1e-7]}]


    N = [2**(i) for i in range(6, 13, 1)] 
    #N.append(107037)

    """
    Try different representations!
    RDKitDescriptors is the best one
    #best model at 0.19 eV for the atomization energy with gamma = 5.96*1e-8 for N = 32768 for RDKitDescriptors
    note atomization energy is  here not normalized by atomic size (following the usual definition).
    for MC with different molecule sizes, set normalize=True in atomization_en()
    """

    DESCRIPTORS = [dc.feat.RDKitDescriptors(ipc_avg=True).featurize(SMILES),get_all_FP(SMILES, fp_type="MorganFingerprint"),get_all_FP(SMILES, fp_type="Avalon")]


    for X in DESCRIPTORS:

        X_train, X_test, y_train, y_test = train_test_split(X,   y, random_state=1337, test_size=0.20, shuffle=True)
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test  = scaler.transform(y_test.reshape(-1,1)).flatten()
        scalerfile = 'scaler.sav'
        pickle.dump(scaler, open(scalerfile, 'wb'))
        lrn_crv      = []
        for n in N:

            clf = Pipeline([('krr', KernelRidge(kernel="laplacian"))])
            grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True, verbose=0, n_jobs=8, refit=True)
            grid_search.fit(X_train[:n], y_train[:n])
            best_model = grid_search.best_estimator_
            predictions     = best_model.predict(X_test)
            MAE             = metrics.mean_absolute_error(scaler.inverse_transform(predictions.reshape(-1,1)).flatten(), scaler.inverse_transform(y_test.reshape(-1,1)).flatten())
            lrn_crv.append(MAE)
            print(n, MAE*27, best_model)

            #model_name = "ATOMIZATION_NORM_QM9_FEATURES_{}".format(n)
            #model_name = "GAP_QM9_FEATURES_{}".format(n)
            #pickle.dump(best_model, open("./ml_data/"+model_name, 'wb'))

    exit()
    scalerfile = 'scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    model = pickle.load(open(model_name, 'rb'))
    predictions = scaler.inverse_transform(model.predict(X_test))
