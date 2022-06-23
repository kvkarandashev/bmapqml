from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import MolToSmiles
from g2s.constants import periodic_table
try:
    from xyz2mol import AC2BO, xyz2AC, BO2mol, chiral_stereo_check, AC2mol, int_atom, str_atom
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install xyz2mol software in order to use this parser. '
                      'Visit: https://github.com/jensengroup/xyz2mol')

class NotConvergedMMFFConformer(Exception):
    pass

from .ext_graph_compound import ExtGraphCompound
from .modify import replace_heavy_atom, atom_replacement_possibilities
import numpy as np
from igraph import Graph
from ..python_parallelization import default_num_procs
from ..utils import canonical_atomtype, read_xyz_file
import os, pickle, copy, tarfile
from sortedcontainers import SortedList

default_parallel_backend="multiprocessing"


def xyz_list2mols_extgraph(xyz_file_list, leave_nones=False, xyz_to_add_data=False):
    read_xyzs=[read_xyz_file(xyz_file) for xyz_file in xyz_file_list]
    unfiltered_list=Parallel(n_jobs=default_num_procs(), backend=default_parallel_backend)(delayed(try_xyz2mol_extgraph)(None, read_xyz=read_xyz) for read_xyz in read_xyzs)
    output=[]
    for egc_id, (egc, xyz_name) in enumerate(zip(unfiltered_list, xyz_file_list)):
        if egc is None:
            print("WARNING, failed to create EGC for id", egc_id, "xyz name:", xyz_name)
        else:
            if xyz_to_add_data:
                egc.additional_data["xyz"]=xyz_name
        if ((egc is not None) or leave_nones):
            output.append(egc)
    return output

def try_xyz2mol_extgraph(xyz_file, read_xyz=None):
#   TODO add exception handling here.
#    try:
        return xyz2mol_extgraph(xyz_file, read_xyz=read_xyz)
#    except:
#        return None

def break_into_connected(egc):
    output=[]
    gc=egc.chemgraph.graph.components()
    mvec=np.array(gc.membership)
    sgcs=gc.subgraphs()

    if egc.distances is None:
        new_dist_mat=None
    if egc.coordinates is None:
        new_coords=None

    adjmat=egc.true_adjmat()

    hids=[[] for i in range(len(sgcs))]
    for h_id in range(egc.num_heavy_atoms(), egc.num_atoms()):
        for ha_id in range(egc.num_heavy_atoms()):
            if adjmat[h_id, ha_id]==1:
                hids[mvec[ha_id]].append(h_id)
                break

    for sgc_id, sgc in enumerate(sgcs):
        new_members=np.where(mvec==sgc_id)[0]
        if len(hids[sgc_id])!=0:
            new_members=np.append(new_members, hids[sgc_id])
        if egc.distances is not None:
            new_dist_mat=np.copy(egc.distances[new_members, :][:, new_members])
        if egc.coordinates is not None:
            new_coords=np.copy(egc.coordinates[new_members, :])
        new_nuclear_charges=np.copy(egc.true_ncharges()[new_members])
        new_adjacency_matrix=np.copy(adjmat[new_members, :][:, new_members])
        output.append(ExtGraphCompound(adjacency_matrix=new_adjacency_matrix, distances=new_dist_mat, nuclear_charges=new_nuclear_charges, coordinates=new_coords))
    return output

def generate_unrepeated_database(egc_list):
    output=SortedList()
    for egc in egc_list:
        if egc not in output:
            output.add(egc)
    return output

def generate_carbonized_unrepeated_database(egc_list, target_el="C"):
    carbon_only_egc_list=[]
    other_egc_list=[]
    target_ncharge=int_atom(target_el)
    for egc in egc_list:
        if other_than_target_el_present(egc, target_el=target_el):
            other_egc_list.append(egc)
        else:
            carbon_only_egc_list.append(egc)
    print("Carbon only molecules:", len(carbon_only_egc_list))
    print("Other molecules:", len(other_egc_list))
    output=generate_unrepeated_database(carbon_only_egc_list)
    print("Nonrepeating carbon molecules:", len(output))
    other_carb_norm_tuples=Parallel(n_jobs=default_num_procs(), backend=default_parallel_backend)(delayed(carb_norm_tuple)(egc, target_el=target_el) for egc in other_egc_list)

    other_carbonated_egc=SortedList()

    for cegc, egc in other_carb_norm_tuples:
        print("Carbonized:", egc)
        if ((cegc not in output) and (egc not in output) and (cegc not in other_carbonated_egc)):
            output.add(egc)
            other_carbonated_egc.add(cegc)
    print("Final number of molecules:", len(output))
    return output

def carb_norm_tuple(egc, target_el="C"):
    return (replace_heavy_wcarbons(egc, target_el=target_el), egc)

def other_than_target_el_present(egc, target_el="C"):
    target_ncharge=int_atom(target_el)
    for i, charge in enumerate(egc.true_ncharges()):
        if ((charge != 1) and (charge != target_ncharge)):
            return True
    return False

def brute_replace_heavy_wcarbons(egc, target_ncharge=6):
    new_charges=egc.true_ncharges()
    for i, ncharge in enumerate(new_charges):
        if ncharge>1:
            new_charges[i]=target_ncharge
    return ExtGraphCompound(adjacency_matrix=egc.true_adjmat(), nuclear_charges=new_charges,
                                coordinates=egc.coordinates, distances=egc.distances)

def replace_heavy_wcarbons(egc, target_el="C"):
    target_charge=int_atom(target_el)
    egc_mod=copy.deepcopy(egc)
    repl_pos=atom_replacement_possibilities(egc_mod, target_el)
    while len(repl_pos)>0:
        egc_mod=replace_heavy_atom(egc_mod, repl_pos[0], target_el)
        repl_pos=atom_replacement_possibilities(egc_mod, target_el)
    return egc_mod

def xyz2mol_graph(nuclear_charges, charge, coords, get_chirality=False):
    _adj_matrix, mol = xyz2AC(nuclear_charges, coords, charge)
    bond_order_matrix, atomic_valence_electrons = AC2BO(_adj_matrix, nuclear_charges, charge,
                                                 allow_charged_fragments=True,
                                                 use_graph=True)
    if get_chirality is False:
        return bond_order_matrix, nuclear_charges, coords
    else:
        chiral_mol = AC2mol(mol, _adj_matrix, nuclear_charges, charge,
                            allow_charged_fragments=True,
                            use_graph=True)
        chiral_stereo_check(chiral_mol)
        chiral_centers = Chem.FindMolChiralCenters(chiral_mol)
        return bond_order_matrix, nuclear_charges, coords, chiral_centers


def xyz2mol_extgraph(filepath, get_chirality=False, read_xyz=None):
    if read_xyz is None:
        read_xyz=read_xyz_file(filepath)

    # Convert numpy array to lists as that is the correct input for xyz2mol_graph function.
    nuclear_charges=[int(ncharge) for ncharge in read_xyz[0]]
    coordinates=[[float(atom_coord) for atom_coord in atom_coords] for atom_coords in read_xyz[2]]
    add_attr_dict=read_xyz[3]

    charge=None
    if "charge" in add_attr_dict:
        charge=add_attr_dict["charge"]
    if charge is None:
        charge=0

    bond_order_matrix, ncharges, coords=xyz2mol_graph(nuclear_charges, charge, coordinates)
    return ExtGraphCompound(adjacency_matrix=bond_order_matrix, nuclear_charges=ncharges, coordinates=np.array(coords))


# TO-DO: should be fixed if I get to using xbgf again.
def xbgf2gc(xbgf_file):
    xbgf_input=open(xbgf_file, 'r')
    graph=Graph()
    while True:
        line=xbgf_input.readline()
        if not line:
            break
        lsplit=line.split()
        if (lsplit[0] == "REMARK") and (lsplit[1]=="NATOM"):
            natom=int(lsplit[2])
            nuclear_charges=np.empty((natom,))
            graph.add_vertices(natom)
        if lsplit[0] == "ATOM":
            atom_id=int(lsplit[1])-1
            nuclear_charges[atom_id]=int(lsplit[15])
        if lsplit[0] == "CONECT":
            if len(lsplit)>2:
                atom_id=int(lsplit[1])-1
                for con_id_str in lsplit[2:]:
                    con_id=int(con_id_str)-1
                    graph.add_edges([(atom_id, con_id)])
    xbgf_input.close()
    return ExtGraphCompound(graph=graph, nuclear_charges=nuclear_charges)

#   Some procedures that often appear in scripts.
def str_atom_corr(ncharge):
    return canonical_atomtype(str_atom(ncharge))

def write_egc2xyz(egc, xyz_file_name):
    write_xyz_file(egc.coordinates, xyz_file_name, charges=egc.nuclear_charges)


def write_xyz_file(coordinates, xyz_file_name, charges=None, elements=None):
    if elements is None:
        elements=[str_atom_corr(charge) for charge in charges]
    xyz_file=open(xyz_file_name, 'w')
    xyz_file.write(str(len(coordinates))+'\n\n')
    for atom_coords, element in zip(coordinates, elements):
        xyz_file.write(element+" "+' '.join([str(atom_coord) for atom_coord in atom_coords])+'\n')
    xyz_file.close()


def all_egc_from_tar(tarfile_name):
    tar_input=tarfile.open(tarfile_name, 'r')
    output=[]
    for tar_member in tar_input.getmembers():
        extracted_member=tar_input.extractfile(tar_member)
        if extracted_member is not None:
            output.append(xyz2mol_extgraph(extracted_member))
    tar_input.close()
    return output

# All the RDKit-related stuff.

#   For going between rdkit and egc objects.
def rdkit_to_egc(rdkit_mol):
    nuclear_charges = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    adjacency_matrix=GetAdjacencyMatrix(rdkit_mol)
    try:
        coordinates=rdkit_mol.GetConformer().GetPositions()
    except ValueError:
        coordinates=None
    return ExtGraphCompound(adjacency_matrix=adjacency_matrix, nuclear_charges=nuclear_charges, coordinates=coordinates)

#   For converting SMILES to egc.
def SMILES_to_egc(smiles_string):
    mol=Chem.MolFromSmiles(smiles_string)
    mol=Chem.AddHs(mol, explicitOnly=True)
    return rdkit_to_egc(mol)

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
            if (iy <= ix) or (iy>=egc.num_atoms()):
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


rdkit_bond_type={1 : Chem.rdchem.BondType.SINGLE, 2 : Chem.rdchem.BondType.DOUBLE, 3 : Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.QUADRUPLE}

def chemgraph_to_canonical_rdkit(cg):
    # create empty editable mol object
    mol = Chem.RWMol()

    inv_canonical_ordering=cg.get_inv_canonical_permutation()
    heavy_atom_index={}
    hydrogen_connection={}
    for atom_counter, atom_id in enumerate(inv_canonical_ordering):
        a = Chem.Atom(periodic_table[cg.hatoms[atom_id].ncharge])
        mol_idx=mol.AddAtom(a)
        heavy_atom_index[atom_id]=mol_idx
        for other_atom_id in inv_canonical_ordering[:atom_counter]:
            bond_order=cg.bond_order(atom_id, other_atom_id)
            if bond_order != 0:
                bond_type=rdkit_bond_type[bond_order]
                mol.AddBond(mol_idx, heavy_atom_index[other_atom_id], bond_type)

    canon_SMILES=MolToSmiles(mol)

    for atom_id in inv_canonical_ordering:
        mol_idx=heavy_atom_index[atom_id]
        for hydrogen_counter in range(cg.hatoms[atom_id].nhydrogens):
            h = Chem.Atom(1)
            mol_hydrogen_idx=mol.AddAtom(h)
            hydrogen_connection[mol_hydrogen_idx]=atom_id
            # Add bond between hydrogen and heavy atom.
            mol.AddBond(mol_idx, mol_hydrogen_idx, rdkit_bond_type[1])

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # TO-DO: Do we need to sanitize?
    Chem.SanitizeMol(mol)
    return mol, heavy_atom_index, hydrogen_connection, canon_SMILES

def egc_with_coords(egc, coords=None, methods="MMFF"):
    output=copy.deepcopy(egc)
    cur_rdkit, heavy_atom_index, hydrogen_connection, canon_SMILES=chemgraph_to_canonical_rdkit(egc.chemgraph)
    if coords is None:
        AllChem.EmbedMolecule(cur_rdkit)
        converged=AllChem.MMFFOptimizeMolecule(cur_rdkit)
        if converged != 0:
            raise NotConvergedMMFFConformer
        coords=np.array(cur_rdkit.GetConformer().GetPositions())

    output.additional_data["canon_rdkit_heavy_atom_index"]=heavy_atom_index
    output.additional_data["canon_rdkit_hydrogen_connection"]=hydrogen_connection
    output.additional_data["canon_rdkit_SMILES"]=canon_SMILES
    output.add_canon_rdkit_coords(coords)
    return output
