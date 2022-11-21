from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import MolToSmiles
from g2s.constants import periodic_table
from .ext_graph_compound import ExtGraphCompound
import copy


class RdKitFailure(Exception):
    pass


#   For going between rdkit and egc objects.
def rdkit_to_egc(rdkit_mol, egc_hydrogen_autofill=False):
    nuclear_charges = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    adjacency_matrix = GetAdjacencyMatrix(rdkit_mol)

    try:
        coordinates = rdkit_mol.GetConformer().GetPositions()
    except ValueError:
        coordinates = None
    return ExtGraphCompound(
        adjacency_matrix=adjacency_matrix,
        nuclear_charges=nuclear_charges,
        coordinates=coordinates,
        hydrogen_autofill=egc_hydrogen_autofill,
    )


#   For converting SMILES to egc.
def SMILES_to_egc(smiles_string, egc_hydrogen_autofill=False):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise RdKitFailure
    # We can fill the hydrogens either at this stage or at EGC creation stage;
    # introduced when I had problems with rdKIT.
    mol = Chem.AddHs(mol, explicitOnly=egc_hydrogen_autofill)
    egc_out = rdkit_to_egc(mol, egc_hydrogen_autofill=egc_hydrogen_autofill)
    egc_out.additional_data["SMILES"] = smiles_string
    return egc_out


def SMILES_list_to_egc(smiles_list):
    return [SMILES_to_egc(smiles) for smiles in smiles_list]


#   For converting InChI to egc.
def InChI_to_egc(InChI_string, egc_hydrogen_autofill=False):
    mol = Chem.inchi.MolFromInchi(InChI_string, removeHs=False)
    if mol is None:
        raise RdKitFailure
    mol = Chem.AddHs(mol, explicitOnly=egc_hydrogen_autofill)
    return rdkit_to_egc(mol, egc_hydrogen_autofill=egc_hydrogen_autofill)


# TO-DO in later stages combine with graph_to_rdkit function in G2S.
def egc_to_rdkit(egc):
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for atom_id, atom_ncharge in enumerate(egc.true_ncharges()):
        a = Chem.Atom(periodic_table[atom_ncharge])
        mol_idx = mol.AddAtom(a)
        node_to_idx[atom_id] = mol_idx

    # add bonds between adjacent atoms
    for ix, row in enumerate(egc.true_adjmat()):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if (iy <= ix) or (iy >= egc.num_atoms()):
                continue
            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond >= 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # TO-DO: Do we need to sanitize?
    Chem.SanitizeMol(mol)
    return mol


rdkit_bond_type = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.QUADRUPLE,
}


def chemgraph_to_canonical_rdkit(cg, SMILES_only=False):
    # create empty editable mol object
    mol = Chem.RWMol()

    inv_canonical_ordering = cg.get_inv_canonical_permutation()
    heavy_atom_index = {}
    hydrogen_connection = {}

    for atom_counter, atom_id in enumerate(inv_canonical_ordering):
        a = Chem.Atom(periodic_table[cg.hatoms[atom_id].ncharge])
        mol_idx = mol.AddAtom(a)
        heavy_atom_index[atom_id] = mol_idx
        for other_atom_id in inv_canonical_ordering[:atom_counter]:
            bond_order = cg.bond_order(atom_id, other_atom_id)
            if bond_order != 0:
                bond_type = rdkit_bond_type[bond_order]
                mol.AddBond(mol_idx, heavy_atom_index[other_atom_id], bond_type)

    canon_SMILES = MolToSmiles(mol)

    if SMILES_only:
        return canon_SMILES

    for atom_id in inv_canonical_ordering:
        mol_idx = heavy_atom_index[atom_id]
        for _ in range(cg.hatoms[atom_id].nhydrogens):
            h = Chem.Atom(1)
            mol_hydrogen_idx = mol.AddAtom(h)
            hydrogen_connection[mol_hydrogen_idx] = atom_id
            # Add bond between hydrogen and heavy atom.
            mol.AddBond(mol_idx, mol_hydrogen_idx, rdkit_bond_type[1])

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # TODO: Do we need to sanitize?
    Chem.SanitizeMol(mol)
    return mol, heavy_atom_index, hydrogen_connection, canon_SMILES


# Different optimizers available for rdkit.
class FFInconsistent(Exception):
    pass


rdkit_coord_optimizer = {
    "MMFF": AllChem.MMFFOptimizeMolecule,
    "UFF": AllChem.UFFOptimizeMolecule,
}


def RDKit_FF_optimize_coords(
    mol, coord_optimizer, num_attempts=1, corresponding_cg=None
):
    try:
        AllChem.EmbedMolecule(mol)
    except:
        print("#PROBLEMATIC_EMBED_MOLECULE:", corresponding_cg)
        raise FFInconsistent
    for _ in range(num_attempts):
        try:
            converged = coord_optimizer(mol)
        except ValueError:
            raise FFInconsistent
        if converged != 1:
            return
    raise FFInconsistent


rdkit_ff_creator = {
    "MMFF": AllChem.MMFFGetMoleculeForceField,
    "UFF": AllChem.UFFGetMoleculeForceField,
}

rdkit_properties_creator = {"MMFF": Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties}


def RDKit_FF_min_en_conf(mol, ff_type, num_attempts=1, corresponding_cg=None):
    """
    Repeats FF coordinate optimization several times to make sure the used configuration is the smallest one.
    """

    min_en = None
    min_coords = None
    min_nuclear_charges = None
    for seed in range(num_attempts):
        cur_mol = copy.deepcopy(mol)

        try:
            AllChem.EmbedMolecule(cur_mol, randomSeed=seed)
        except:
            print("#PROBLEMATIC_EMBED_MOLECULE:", corresponding_cg)

        args = (cur_mol,)

        if ff_type in rdkit_properties_creator:
            prop_obj = rdkit_properties_creator[ff_type](cur_mol)
            args = (*args, prop_obj)
        try:
            ff = rdkit_ff_creator[ff_type](*args)
        except ValueError:
            cur_en = None
        try:
            converted = ff.Minimize()
            cur_en = ff.CalcEnergy()
        except:
            raise FFInconsistent
        if converted != 0:
            continue
        try:
            cur_coords = np.array(np.array(cur_mol.GetConformer().GetPositions()))
            cur_nuclear_charges = np.array(
                [atom.GetAtomicNum() for atom in cur_mol.GetAtoms()]
            )
        except ValueError:
            cur_coords = None
            cur_nuclear_charges = None
        if ((cur_en is not None) and (cur_coords is not None)) and (
            (min_en is None) or (min_en > cur_en)
        ):
            min_en = cur_en
            min_coords = cur_coords
            min_nuclear_charges = cur_nuclear_charges

    if min_en is None:
        raise FFInconsistent

    return min_coords, min_nuclear_charges, min_en
