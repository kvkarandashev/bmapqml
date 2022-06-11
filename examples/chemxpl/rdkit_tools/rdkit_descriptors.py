import os
from tqdm import tqdm
from rdkit.Chem import DataStructs
import rdkit
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Lipinski import HeavyAtomCount,NumHAcceptors, NumHeteroatoms,NumRotatableBonds, NHOHCount, NOCount, NumHDonors, RingCount, NumSaturatedHeterocycles, NumSaturatedRings, FractionCSP3
from rdkit.Chem.Descriptors import MaxPartialCharge, MinPartialCharge, MinAbsPartialCharge, MaxAbsPartialCharge,ExactMolWt,MolWt,FpDensityMorgan1,FpDensityMorgan2,FpDensityMorgan3,NumRadicalElectrons,NumValenceElectrons
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit import Chem  
import numpy as np
import collections

def canonize(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

def count_element(atoms):

    """
    Returns the number of each element in the molecule
    """

    comp = collections.Counter(atoms)

    return np.int_(np.array([comp[1], comp[6], comp[7],comp[8],comp[9]]))


def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def COMPQ_PLUS(smi, padding=50):

    """
    Computes the COMPQ+ fingerprint of a molecule given its SMILES. 
    This is only based on rdkit features but outperforms the MORGAN fingerprint.
    This is very ugly code but it works.
    """
    
    mol = Chem.MolFromSmiles(canonize(smi))
    mol = rdkit.Chem.rdmolops.AddHs(mol)
    hdn = NumHDonors(mol)
    hacc = HeavyAtomCount(mol)
    hacc2 = RingCount(mol)

    hacc3 = NumSaturatedHeterocycles(mol)
    hacc4 = NumSaturatedRings(mol)
    hacc5 = MaxPartialCharge(mol)
    hacc6 = MinPartialCharge(mol)
    hacc8 = NumHAcceptors(mol)
    hacc9 = NumHeteroatoms(mol)
    hacc10 = NumRotatableBonds(mol)
    hacc11 = NHOHCount(mol)
    hacc12 = NOCount(mol)
    hacc13 = FractionCSP3(mol)


    nuclear_charges = np.array([x.GetAtomicNum() for x in mol.GetAtoms()])
    q = count_element(nuclear_charges)
    q = np.append(q,hdn)
    q = np.append(q,hacc)
    q = np.append(q,hacc2)
    q = np.append(q,hacc3)
    q = np.append(q,hacc4)
    q = np.append(q,hacc5)
    q = np.append(q,hacc6)
    q = np.append(q,hacc8)
    q = np.append(q,hacc9)
    q = np.append(q,hacc10)
    q = np.append(q,hacc11)
    q = np.append(q,hacc12)
    q = np.append(q,hacc13)
    q = np.append(q,NumSaturatedHeterocycles(mol))
    q = np.append(q,NumSaturatedRings(mol))
    q = np.append(q,FractionCSP3(mol))
    q = np.append(q,MinAbsPartialCharge(mol))
    q = np.append(q,MaxAbsPartialCharge(mol))
    q = np.append(q,ExactMolWt(mol))
    q = np.append(q,MolWt(mol))
    q = np.append(q,FpDensityMorgan1(mol))
    q = np.append(q,FpDensityMorgan2(mol))
    q = np.append(q,FpDensityMorgan3(mol))
    q = np.append(q,NumRadicalElectrons(mol))
    q = np.append(q,NumValenceElectrons(mol))
    q = np.append(q,MolLogP(mol))
    q = np.append(q,MolMR(mol))
    
    
    q = padarray(q, padding)
    
    return q


def ExplicitBitVect_to_NumpyArray(fp_vec):

    """
    Convert the rdkit fingerprint to a numpy array
    """

    fp2 = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp_vec, fp2)
    return fp2


def get_single_FP(smi, fp_type):
    
    """
    Computes the fingerprint of a molecule given its SMILES
    Input:
    smi: SMILES string
    fp_type: type of fingerprint to be computed
    """

    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smi)

    if fp_type=="MorganFingerprint":
        fp_mol = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=4,
            nBits=8192,
            useFeatures=True,
        )

    return fp_mol

def get_all_FP(SMILES, fp_type):

    """
    Returns a list of fingerprints for all the molecules in the list of SMILES
    """

    X = []
    for smi in SMILES:
        if fp_type=="MorganFingerprint":
            x = ExplicitBitVect_to_NumpyArray(get_single_FP(smi, "MorganFingerprint"))
            X.append(x)
        if fp_type=="COMPQ":

            x = COMPQ_PLUS(smi)
            X.append(x)

        if fp_type=="both":

            x1  = ExplicitBitVect_to_NumpyArray(get_single_FP(smi, "MorganFingerprint"))
            x2  = COMPQ_PLUS(smi)
            x   = np.concatenate((x1, x2))
            X.append(x)

    X = np.array(X)

    return X




def atomization_en(EN, ATOMS, normalize=True):

    """
    Compute the atomization energy, if normalize is True, 
    the output is normalized by the number of atoms. This allows 
    predictions to be consistent when comparing molecules of different size
    with respect to their bond energies i.e. set to True if the number of atoms 
    changes in during the optimization process
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

def process_qm9(directory):
    
    """
    Reads the xyz files in the directory on 'path' as well as the properties of 
    the molecules in the same directory.
    """


    file = os.listdir(directory)[0]

    data = []
    smiles = []
    properties = []
    for file in tqdm(os.listdir(directory)[:133884]  ): #):
        path = os.path.join(directory, file)
        atoms, coordinates, smile, prop = read_xyz(path)
        # A tuple with the atoms and its coordinates
        data.append((atoms, coordinates))
        smiles.append(smile)  # The SMILES representation

        ATOMIZATION = atomization_en(float(prop[10]), atoms, normalize=True)
        prop += [ATOMIZATION]
        properties.append(prop)  # The molecules properties

    properties_names = ['A', 'B', 'C', 'mu', 'alfa', 'homo', 'lumo', 'gap', 'R²', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'atomization']
    df = pd.DataFrame(properties, columns = properties_names) #.astype('float32')
    df['smiles'] = smiles
    df.head()

    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['mol'].isnull().sum()

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


def mae(prediction, reference):

    """
    Compute the MAE between prediction and
    reference data
    """

    return np.mean(np.abs(prediction - reference))
