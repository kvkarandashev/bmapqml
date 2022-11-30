from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import MolToSmiles
from g2s.constants import periodic_table
from .valence_treatment import ChemGraph, default_valence
from .ext_graph_compound import ExtGraphCompound
import copy
import numpy as np


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


rdkit_bond_type = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.QUADRUPLE,
}

# TODO in later stages combine with graph_to_rdkit function in G2S.
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
            else:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit_bond_type[bond])

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # TODO: Do we need to sanitize?
    Chem.SanitizeMol(mol)
    return mol


def chemgraph_to_rdkit(
    cg: ChemGraph,
    explicit_hydrogens=True,
    resonance_struct_adj=None,
    extra_valence_hydrogens=False,
    get_rw_mol=False,
):
    """
    Create an rdkit mol object from a ChemGraph object.
    """
    # create empty editable mol object
    mol = Chem.RWMol()
    nhydrogens = np.zeros((cg.nhatoms(),), dtype=int)

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for atom_id, ha in enumerate(cg.hatoms):
        a = Chem.Atom(periodic_table[ha.ncharge])
        mol_idx = mol.AddAtom(a)
        node_to_idx[atom_id] = mol_idx
        nhydrogens[atom_id] = ha.nhydrogens

    # add bonds between adjacent atoms
    for ix in range(cg.nhatoms()):
        for iy in cg.neighbors(ix):
            if iy < ix:
                continue
            btuple = (ix, iy)
            bo = cg.bond_orders[btuple]
            if resonance_struct_adj is not None:
                if btuple in cg.resonance_structure_map:
                    res_struct_id = cg.resonance_structure_map[btuple]
                    if res_struct_id in resonance_struct_adj:
                        bo = cg.aa_all_bond_orders(*btuple, unsorted=True)[
                            resonance_struct_adj[res_struct_id]
                        ]
            # add relevant bond type (there are many more of these)
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit_bond_type[bo])

    # TODO Didn't I have a DEFAULT_ATOM somewhere?
    if explicit_hydrogens:
        for ha_id, nhyd in enumerate(nhydrogens):
            for _ in range(nhyd):
                a = Chem.Atom(1)
                hidx = mol.AddAtom(a)
                mol.AddBond(node_to_idx[ha_id], hidx, rdkit_bond_type[1])
    elif extra_valence_hydrogens:
        for ha_id, ha in enumerate(cg.hatoms):
            if (resonance_struct_adj is None) or (ha.possible_valences is None):
                cur_valence = ha.valence
            else:
                # TODO do we need a function for finding which resonance structure contains a given atom?
                for i, extra_val_ids in enumerate(cg.resonance_structure_inverse_map):
                    if ha_id in extra_val_ids:
                        cur_valence = ha.possible_valences[
                            cg.resonance_structure_valence_vals[i][
                                resonance_struct_adj[res_struct_id]
                            ]
                        ]
            if cur_valence != default_valence(ha.ncharge):
                for _ in range(ha.nhydrogens):
                    a = Chem.Atom(1)
                    hidx = mol.AddAtom(a)
                    mol.AddBond(node_to_idx[ha_id], hidx, rdkit_bond_type[1])

    if not get_rw_mol:
        # Convert RWMol to Mol object
        mol = mol.GetMol()
        # TODO: Do we need to sanitize?
        Chem.SanitizeMol(mol)
    return mol


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
