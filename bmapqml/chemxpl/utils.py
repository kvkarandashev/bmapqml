from joblib import Parallel, delayed

try:
    from xyz2mol import (
        AC2BO,
        xyz2AC,
        chiral_stereo_check,
        AC2mol,
        int_atom,
    )
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Install xyz2mol software in order to use this parser. "
        "Visit: https://github.com/jensengroup/xyz2mol"
    )

from .ext_graph_compound import ExtGraphCompound
from .modify import replace_heavy_atom, atom_replacement_possibilities
from .rdkit_utils import *
import numpy as np
from igraph import Graph
from ..utils import read_xyz_file, default_num_procs, write_xyz_file, xyz_string
from .valence_treatment import ChemGraph, InvalidAdjMat, str2ChemGraph
import copy, tarfile
from sortedcontainers import SortedList

default_parallel_backend = "multiprocessing"

default_minfunc_name = "MIN_FUNC_NAME"


def xyz_list2mols_extgraph(xyz_file_list, leave_nones=False, xyz_to_add_data=False):
    read_xyzs = [read_xyz_file(xyz_file) for xyz_file in xyz_file_list]
    unfiltered_list = Parallel(
        n_jobs=default_num_procs(), backend=default_parallel_backend
    )(delayed(try_xyz2mol_extgraph)(None, read_xyz=read_xyz) for read_xyz in read_xyzs)
    output = []
    for egc_id, (egc, xyz_name) in enumerate(zip(unfiltered_list, xyz_file_list)):
        if egc is None:
            print("WARNING, failed to create EGC for id", egc_id, "xyz name:", xyz_name)
        else:
            if xyz_to_add_data:
                egc.additional_data["xyz"] = xyz_name
        if (egc is not None) or leave_nones:
            output.append(egc)
    return output


def try_xyz2mol_extgraph(xyz_file, read_xyz=None):
    #   TODO add exception handling here.
    #    try:
    return xyz2mol_extgraph(xyz_file, read_xyz=read_xyz)


#    except:
#        return None


def break_into_connected(egc):
    output = []
    gc = egc.chemgraph.graph.components()
    mvec = np.array(gc.membership)
    sgcs = gc.subgraphs()

    if egc.distances is None:
        new_dist_mat = None
    if egc.coordinates is None:
        new_coords = None

    adjmat = egc.true_adjmat()

    hids = [[] for i in range(len(sgcs))]
    for h_id in range(egc.num_heavy_atoms(), egc.num_atoms()):
        for ha_id in range(egc.num_heavy_atoms()):
            if adjmat[h_id, ha_id] == 1:
                hids[mvec[ha_id]].append(h_id)
                break

    for sgc_id, _ in enumerate(sgcs):
        new_members = np.where(mvec == sgc_id)[0]
        if len(hids[sgc_id]) != 0:
            new_members = np.append(new_members, hids[sgc_id])
        if egc.distances is not None:
            new_dist_mat = np.copy(egc.distances[new_members, :][:, new_members])
        if egc.coordinates is not None:
            new_coords = np.copy(egc.coordinates[new_members, :])
        new_nuclear_charges = np.copy(egc.true_ncharges()[new_members])
        new_adjacency_matrix = np.copy(adjmat[new_members, :][:, new_members])
        output.append(
            ExtGraphCompound(
                adjacency_matrix=new_adjacency_matrix,
                distances=new_dist_mat,
                nuclear_charges=new_nuclear_charges,
                coordinates=new_coords,
            )
        )
    return output


def generate_unrepeated_database(egc_list):
    output = SortedList()
    for egc in egc_list:
        if egc not in output:
            output.add(egc)
    return output


def generate_carbonized_unrepeated_database(egc_list, target_el="C"):
    carbon_only_egc_list = []
    other_egc_list = []
    for egc in egc_list:
        if other_than_target_el_present(egc, target_el=target_el):
            other_egc_list.append(egc)
        else:
            carbon_only_egc_list.append(egc)
    print("Carbon only molecules:", len(carbon_only_egc_list))
    print("Other molecules:", len(other_egc_list))
    output = generate_unrepeated_database(carbon_only_egc_list)
    print("Nonrepeating carbon molecules:", len(output))
    other_carb_norm_tuples = Parallel(
        n_jobs=default_num_procs(), backend=default_parallel_backend
    )(delayed(carb_norm_tuple)(egc, target_el=target_el) for egc in other_egc_list)

    other_carbonated_egc = SortedList()

    for cegc, egc in other_carb_norm_tuples:
        print("Carbonized:", egc)
        if (
            (cegc not in output)
            and (egc not in output)
            and (cegc not in other_carbonated_egc)
        ):
            output.add(egc)
            other_carbonated_egc.add(cegc)
    print("Final number of molecules:", len(output))
    return output


def carb_norm_tuple(egc, target_el="C"):
    return (replace_heavy_wcarbons(egc, target_el=target_el), egc)


def other_than_target_el_present(egc, target_el="C"):
    target_ncharge = int_atom(target_el)
    for i, charge in enumerate(egc.true_ncharges()):
        if (charge != 1) and (charge != target_ncharge):
            return True
    return False


def brute_replace_heavy_wcarbons(egc, target_ncharge=6):
    new_charges = egc.true_ncharges()
    for i, ncharge in enumerate(new_charges):
        if ncharge > 1:
            new_charges[i] = target_ncharge
    return ExtGraphCompound(
        adjacency_matrix=egc.true_adjmat(),
        nuclear_charges=new_charges,
        coordinates=egc.coordinates,
        distances=egc.distances,
    )


def replace_heavy_wcarbons(egc, target_el="C"):
    egc_mod = copy.deepcopy(egc)
    repl_pos = atom_replacement_possibilities(egc_mod, target_el)
    while len(repl_pos) > 0:
        egc_mod = replace_heavy_atom(egc_mod, repl_pos[0], target_el)
        repl_pos = atom_replacement_possibilities(egc_mod, target_el)
    return egc_mod


def xyz2mol_graph(nuclear_charges, charge, coords, get_chirality=False):
    try:
        _adj_matrix, mol = xyz2AC(nuclear_charges, coords, charge)
        bond_order_matrix, _ = AC2BO(
            _adj_matrix,
            nuclear_charges,
            charge,
            allow_charged_fragments=True,
            use_graph=True,
        )
    except:
        raise InvalidAdjMat
    if get_chirality is False:
        return bond_order_matrix, nuclear_charges, coords
    else:
        chiral_mol = AC2mol(
            mol,
            _adj_matrix,
            nuclear_charges,
            charge,
            allow_charged_fragments=True,
            use_graph=True,
        )
        chiral_stereo_check(chiral_mol)
        chiral_centers = Chem.FindMolChiralCenters(chiral_mol)
        return bond_order_matrix, nuclear_charges, coords, chiral_centers


def chemgraph_from_ncharges_coords(nuclear_charges, coordinates, charge=0):
    # Convert numpy array to lists as that is the correct input for xyz2mol_graph function.
    converted_ncharges = [int(ncharge) for ncharge in nuclear_charges]
    converted_coordinates = [
        [float(atom_coord) for atom_coord in atom_coords] for atom_coords in coordinates
    ]
    bond_order_matrix, ncharges, _ = xyz2mol_graph(
        converted_ncharges, charge, converted_coordinates
    )
    return ChemGraph(adj_mat=bond_order_matrix, nuclear_charges=ncharges)


def chemgraph_to_canonical_rdkit_wcoords(
    cg, ff_type="MMFF", num_attempts=1, pick_minimal_conf=False
):
    """
    Creates an rdkit Molecule object whose heavy atoms are canonically ordered.
    cg : ChemGraph input chemgraph object
    ff_type : which forcefield to use; currently MMFF and UFF are available
    num_attempts : how many times the optimization is attempted
    output : RDKit molecule, indices of the heavy atoms, indices of heavy atoms to which a given hydrogen is connected,
    SMILES generated from the canonical RDKit molecules, and the RDKit's coordinates
    """
    (
        mol,
        heavy_atom_index,
        hydrogen_connection,
        canon_SMILES,
    ) = chemgraph_to_canonical_rdkit(cg)

    if pick_minimal_conf:
        rdkit_coords, rdkit_nuclear_charges, _ = RDKit_FF_min_en_conf(
            mol, ff_type, num_attempts=num_attempts, corresponding_cg=cg
        )
    else:
        RDKit_FF_optimize_coords(
            mol,
            rdkit_coord_optimizer[ff_type],
            num_attempts=num_attempts,
            corresponding_cg=cg,
        )
        rdkit_coords = np.array(mol.GetConformer().GetPositions())
        rdkit_nuclear_charges = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        )
    # Additionally check that the coordinates actually correspond to the molecule.
    try:
        coord_based_cg = chemgraph_from_ncharges_coords(
            rdkit_nuclear_charges, rdkit_coords
        )
    except InvalidAdjMat:
        raise FFInconsistent
    if coord_based_cg != cg:
        raise FFInconsistent

    return mol, heavy_atom_index, hydrogen_connection, canon_SMILES, rdkit_coords


def egc_from_ncharges_coords(nuclear_charges, coordinates, charge=0):
    cg = chemgraph_from_ncharges_coords(nuclear_charges, coordinates, charge=0)
    return ExtGraphCompound(chemgraph=cg, coordinates=np.array(coordinates))


def xyz2mol_extgraph(filepath, get_chirality=False, read_xyz=None):
    if read_xyz is None:
        read_xyz = read_xyz_file(filepath)

    nuclear_charges = read_xyz[0]
    coordinates = read_xyz[2]
    add_attr_dict = read_xyz[3]

    charge = None
    if "charge" in add_attr_dict:
        charge = add_attr_dict["charge"]
    if charge is None:
        charge = 0

    return egc_from_ncharges_coords(nuclear_charges, coordinates, charge=charge)


# TO-DO: should be fixed if I get to using xbgf again.
def xbgf2gc(xbgf_file):
    xbgf_input = open(xbgf_file, "r")
    graph = Graph()
    while True:
        line = xbgf_input.readline()
        if not line:
            break
        lsplit = line.split()
        if (lsplit[0] == "REMARK") and (lsplit[1] == "NATOM"):
            natom = int(lsplit[2])
            nuclear_charges = np.empty((natom,))
            graph.add_vertices(natom)
        if lsplit[0] == "ATOM":
            atom_id = int(lsplit[1]) - 1
            nuclear_charges[atom_id] = int(lsplit[15])
        if lsplit[0] == "CONECT":
            if len(lsplit) > 2:
                atom_id = int(lsplit[1]) - 1
                for con_id_str in lsplit[2:]:
                    con_id = int(con_id_str) - 1
                    graph.add_edges([(atom_id, con_id)])
    xbgf_input.close()
    return ExtGraphCompound(graph=graph, nuclear_charges=nuclear_charges)


#   Some procedures that often appear in scripts.


def egc2xyz_string(egc, extra_string=""):
    return xyz_string(
        egc.coordinates, nuclear_charges=egc.nuclear_charges, extra_string=extra_string
    )


def write_egc2xyz(egc, xyz_file_name, extra_string=""):
    write_xyz_file(
        egc.coordinates,
        xyz_file_name,
        nuclear_charges=egc.nuclear_charges,
        extra_string=extra_string,
    )


def all_egc_from_tar(tarfile_name):
    tar_input = tarfile.open(tarfile_name, "r")
    output = []
    for tar_member in tar_input.getmembers():
        extracted_member = tar_input.extractfile(tar_member)
        if extracted_member is not None:
            output.append(xyz2mol_extgraph(extracted_member))
    tar_input.close()
    return output


def ChemGraphStr_to_SMILES(chemgraph_str):
    cg = str2ChemGraph(chemgraph_str)
    return chemgraph_to_canonical_rdkit(cg, SMILES_only=True)


def trajectory_point_to_canonical_rdkit(tp_in, SMILES_only=False):
    return chemgraph_to_canonical_rdkit(tp_in.egc.chemgraph, SMILES_only=SMILES_only)


def egc_with_coords(
    egc, coords=None, ff_type="MMFF", num_attempts=1, pick_minimal_conf=False
):
    """
    Create a copy of an ExtGraphCompound object with coordinates. If coordinates are set to None they are generated with RDKit.
    egc : ExtGraphCompound input object
    coords : None or np.array
    ff_type : str type of force field used; MMFF and UFF are available
    num_attempts : int number of
    """
    output = copy.deepcopy(egc)
    if coords is None:
        (
            _,
            heavy_atom_index,
            hydrogen_connection,
            canon_SMILES,
            coords,
        ) = chemgraph_to_canonical_rdkit_wcoords(
            egc.chemgraph,
            ff_type=ff_type,
            num_attempts=num_attempts,
            pick_minimal_conf=pick_minimal_conf,
        )
    else:
        (
            _,
            heavy_atom_index,
            hydrogen_connection,
            canon_SMILES,
        ) = chemgraph_to_canonical_rdkit(egc.chemgraph)

    output.additional_data["canon_rdkit_heavy_atom_index"] = heavy_atom_index
    output.additional_data["canon_rdkit_hydrogen_connection"] = hydrogen_connection
    output.additional_data["canon_rdkit_SMILES"] = canon_SMILES
    output.add_canon_rdkit_coords(coords)
    return output


def coord_info_from_tp(tp, **kwargs):
    """
    Coordinates corresponding to a TrajectoryPoint object
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "canon_rdkit_SMILES": None, "nuclear_charges": None}
    try:
        egc_wc = egc_with_coords(tp.egc, **kwargs)
        output["coordinates"] = egc_wc.coordinates
        output["canon_rdkit_SMILES"] = egc_wc.additional_data["canon_rdkit_SMILES"]
        output["nuclear_charges"] = egc_wc.true_ncharges()
    except FFInconsistent:
        pass

    return output


def canonical_SMILES_from_tp(tp):
    return chemgraph_to_canonical_rdkit(tp.egc.chemgraph, SMILES_only=True)


def bin_index(val, val_lbound=None, val_ubound=None, num_bins=1):
    if val_lbound is None:
        return 0
    else:
        return int(num_bins * (val - val_lbound) / (val_ubound - val_lbound))


def print_distribution_analysis(
    histogram,
    betas,
    min_func_name=default_minfunc_name,
    val_lbound=None,
    val_ubound=None,
    num_bins=1,
):
    """
    Analyse the distribution in the histogram and compare it to Boltzmann weights.
    """
    nmols = np.zeros((num_bins,), dtype=int)
    tot_min = None
    tot_max = None
    for tp in histogram:
        cur_f = tp.calculated_data[min_func_name]
        if (tot_min is None) or (cur_f < tot_min):
            tot_min = cur_f
        if (tot_max is None) or (cur_f > tot_max):
            tot_max = cur_f
        cur_bin_id = bin_index(
            cur_f, val_lbound=val_lbound, val_ubound=val_ubound, num_bins=num_bins
        )
        nmols[cur_bin_id] += 1
    print("Range of minimized function values:", tot_min, tot_max)
    print("TOTAL NUMBERS OF CONSIDERED MOLECULES:")
    for bin_id, nmol in enumerate(nmols):
        print(bin_id, nmol)
    for beta_id, beta in enumerate(betas):
        print("DATA FOR REPLICA", beta_id, "BETA:", beta)
        print_distribution_analysis_single_beta(
            histogram,
            beta_id,
            beta,
            min_func_name,
            val_lbound=val_lbound,
            val_ubound=val_ubound,
            num_bins=num_bins,
        )


def print_distribution_analysis_single_beta(
    histogram,
    beta_id,
    beta,
    min_func_name,
    val_lbound=None,
    val_ubound=None,
    num_bins=1,
):
    """
    Analyse the distribution in the histogram and compare it to Boltzmann weights for one beta.
    histogram : histogram to be analyzed, list of TrajectoryPoint objects
    beta_id : index of the replica to be analyzed
    beta : inverse beta
    min_func_name : name of the minimized function
    num_bins : number of bins into which values are sorted
    val_lbound : upper bound of values to be considered
    val_ubound : lower bound of values to be considered
    """
    if (val_lbound is None) or (val_ubound is None):
        num_bins = 1
    distr = np.zeros((num_bins,))
    distr2 = np.zeros((num_bins,))
    present = np.zeros((num_bins,), dtype=bool)
    nmols = np.zeros((num_bins,), dtype=int)
    bin_middles = np.zeros((num_bins,))
    if val_lbound is None:
        bin_middles[0] = 0.0
    else:
        for i in range(num_bins):
            bin_middles[i] = (
                val_lbound + (i + 0.5) * (val_ubound - val_lbound) / num_bins
            )
    #    bin_middle_boltzmann_weights=np.exp(-beta*bin_middles)
    for tp in histogram:
        cur_f = tp.calculated_data[min_func_name]
        cur_bin_id = bin_index(
            cur_f, val_lbound=val_lbound, val_ubound=val_ubound, num_bins=num_bins
        )
        cur_d = tp.visit_num(beta_id)
        if cur_d > 0.5:
            present[cur_bin_id] = True
        cur_r = cur_d
        if beta is not None:
            cur_r *= np.exp(beta * (cur_f - bin_middles[cur_bin_id]))
        distr[cur_bin_id] += cur_r
        distr2[cur_bin_id] += cur_r**2
        nmols[cur_bin_id] += 1

    stddevs = np.zeros((num_bins,))
    errs = np.zeros((num_bins,))
    rel_errs = np.zeros((num_bins,))
    min_err_id = None
    min_err = None

    for i in np.where(present)[0]:
        distr[i] /= nmols[i]
        distr2[i] /= nmols[i]
        stddevs[i] = np.sqrt(distr2[i] - distr[i] ** 2)
        errs[i] = stddevs[i] / np.sqrt(nmols[i])
        rel_errs[i] = errs[i] / distr[i]
        if (min_err is None) or (min_err > rel_errs[i]) and (nmols[i] != 1):
            min_err = rel_errs[i]
            min_err_id = i

    if min_err_id is None:
        min_err_id = np.argmax(distr)

    print(
        "Distributions: (bin index : adjusted probability density : standard deviation : number of molecules found)"
    )
    for bin_id, (d, stddev, nmol) in enumerate(zip(distr, stddevs, nmols)):
        print(bin_id, d, stddev, nmol)
    if beta is not None:
        print(
            "Deviation of probability densities from Boltzmann factors: (deviation : estimated error)"
        )
        for i in np.where(present)[0]:
            dev = np.log(distr[i] / distr[min_err_id]) + beta * (
                bin_middles[i] - bin_middles[min_err_id]
            )
            if i == min_err_id:
                dev_err = 0.0
            else:
                dev_err = np.sqrt(rel_errs[i] ** 2 + rel_errs[min_err_id] ** 2)
            print(i, dev, dev_err)
