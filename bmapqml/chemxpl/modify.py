# TODO Perhaps add support for atoms with higher valences being added directly?
# TODO check that the currently commenting change_valence function exception is correct

import numpy as np
from .ext_graph_compound import ExtGraphCompound
from .valence_treatment import (
    ChemGraph,
    default_valence,
    avail_val_list,
    int_atom_checked,
    connection_forbidden,
    max_bo,
    sorted_by_membership,
    next_valence,
    sorted_tuple,
)
from copy import deepcopy
import random, itertools
from igraph.operators import disjoint_union
from ..utils import canonical_atomtype
from ..data import NUCLEAR_CHARGE

# TODO delete post-testing
from .rdkit_utils import chemgraph_to_canonical_rdkit


def atom_equivalent_to_list_member(egc, atom_id, atom_id_list):
    if len(atom_id_list) == 0:
        return False
    are_tuples = not isinstance(atom_id_list, dict)
    if are_tuples:
        are_tuples = isinstance(atom_id_list[0], tuple)
    for other_atom_id in atom_id_list:
        if are_tuples:
            true_other_atom_id = other_atom_id[0]
        else:
            true_other_atom_id = other_atom_id
        if egc.chemgraph.atom_pair_equivalent(atom_id, true_other_atom_id):
            return True
    return False


def atom_pair_equivalent_to_list_member(egc, atom_pair, atom_pair_list):
    for other_atom_pair in atom_pair_list:
        if egc.chemgraph.atom_sets_equivalent(atom_pair[:2], other_atom_pair[:2]):
            return True
    return False


def atom_replacement_possibilities(
    egc,
    inserted_atom,
    inserted_valence=None,
    replaced_atom=None,
    forbidden_bonds=None,
    exclude_equivalent=True,
    not_protonated=None,
    default_valences=None,
    **other_kwargs,
):

    possible_ids = []
    inserted_iac = int_atom_checked(inserted_atom)
    if replaced_atom is not None:
        replaced_iac = int_atom_checked(replaced_atom)
    if inserted_valence is None:
        if default_valences is None:
            inserted_valence = default_valences[inserted_iac]
        else:
            inserted_valence = default_valence(inserted_iac)
    if not_protonated is not None:
        cant_be_protonated = inserted_iac in not_protonated
    cg = egc.chemgraph
    for ha_id, ha in enumerate(cg.hatoms):
        if replaced_atom is not None:
            if ha.ncharge != replaced_iac:
                continue
        if inserted_iac == ha.ncharge:
            continue
        if forbidden_bonds is not None:
            cont = False
            for neigh in cg.neighbors(ha_id):
                if connection_forbidden(
                    egc.nuclear_charges[neigh], inserted_atom, forbidden_bonds
                ):
                    cont = True
                    break
            if cont:
                continue

        ha_default_valence = default_valence(ha.ncharge)

        val_diff = ha_default_valence - inserted_valence
        if val_diff <= ha.nhydrogens:
            if not_protonated is not None:
                if cant_be_protonated and (val_diff != ha.nhydrogens):
                    continue

            if ha.possible_valences is None:
                if ha_default_valence == ha.valence:
                    resonance_structure_id = None
                else:
                    continue
            else:
                if ha_default_valence in ha.possible_valences:
                    resonance_structure_id = cg.atom_valence_resonance_structure_id(
                        hatom_id=ha_id, valence=ha_default_valence
                    )
                else:
                    continue

            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            if ha.possible_valences is None:
                if ha_default_valence == ha.valence:
                    resonance_structure_id = None
                else:
                    continue
            else:
                if ha_default_valence in ha.possible_valences:
                    resonance_structure_id = cg.atom_valence_resonance_structure_id(
                        hatom_id=ha_id, valence=ha_default_valence
                    )
                else:
                    continue
            possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


def gen_atom_removal_possible_hnums(added_bond_orders, default_valence):
    possible_hnums = []
    for abo in added_bond_orders:
        hnum = default_valence - abo
        if hnum >= 0:
            possible_hnums.append(hnum)
    return possible_hnums


def atom_removal_possibilities(
    egc,
    deleted_atom="C",
    exclude_equivalent=True,
    nhatoms_range=None,
    not_protonated=None,
    added_bond_orders=[1],
    default_valences=None,
    atom_removal_possible_hnums=None,
    **other_kwargs,
):
    if nhatoms_range is not None:
        if egc.num_heavy_atoms() <= nhatoms_range[0]:
            return []
    possible_ids = []
    deleted_iac = int_atom_checked(deleted_atom)
    if default_valences is not None:
        deleted_default_valence = default_valences[deleted_iac]
    else:
        deleted_default_valence = default_valence(deleted_iac)

    if atom_removal_possible_hnums is None:
        possible_hnums = gen_atom_removal_possible_hnums(
            added_bond_orders, deleted_default_valence
        )
    else:
        possible_hnums = atom_removal_possible_hnums[deleted_iac]

    cg = egc.chemgraph
    hatoms = cg.hatoms

    for ha_id, ha in enumerate(hatoms):
        if (ha.ncharge != deleted_iac) or (ha.nhydrogens not in possible_hnums):
            continue
        neighs = cg.neighbors(ha_id)
        if len(neighs) != 1:
            continue
        if not_protonated is not None:
            if hatoms[neighs[0]].ncharge in not_protonated:
                continue
        if ha.possible_valences is None:
            if ha.valence != deleted_default_valence:
                continue
            resonance_structure_id = None
        else:
            if deleted_default_valence in ha.possible_valences:
                resonance_structure_id = cg.atom_valence_resonance_structure_id(
                    hatom_id=ha_id, valence=deleted_default_valence
                )
            else:
                continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        possible_ids.append((ha_id, resonance_structure_id))
    return possible_ids


def chain_addition_possibilities(
    egc,
    chain_starting_element=None,
    forbidden_bonds=None,
    exclude_equivalent=True,
    nhatoms_range=None,
    not_protonated=None,
    added_bond_orders=[1],
    avail_added_bond_orders=None,
    chain_addition_tuple_possibilities=False,
    **other_kwargs,
):
    if chain_addition_tuple_possibilities:
        possible_ids = []
    else:
        possible_ids = {}

    if nhatoms_range is not None:
        if egc.num_heavy_atoms() >= nhatoms_range[1]:
            return possible_ids
    chain_starting_ncharge = int_atom_checked(chain_starting_element)

    if avail_added_bond_orders is None:
        avail_added_bond_order = available_added_atom_bos(
            chain_starting_ncharge, added_bond_orders, not_protonated=not_protonated
        )
    else:
        avail_added_bond_order = avail_added_bond_orders[chain_starting_ncharge]

    if len(avail_added_bond_order) == 0:
        return possible_ids

    min_avail_added_bond_order = min(avail_added_bond_order)

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if (ha.nhydrogens >= min_avail_added_bond_order) and (
            not connection_forbidden(
                ha.ncharge, chain_starting_ncharge, forbidden_bonds
            )
        ):
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            for added_bond_order in avail_added_bond_order:
                if added_bond_order > max_bo(ha, chain_starting_ncharge):
                    continue
                if added_bond_order <= ha.nhydrogens:
                    if chain_addition_tuple_possibilities:
                        possible_ids.append((ha_id, added_bond_order))
                    else:
                        if ha_id in possible_ids:
                            possible_ids[ha_id].append(added_bond_order)
                        else:
                            possible_ids[ha_id] = [added_bond_order]
    return possible_ids


def bond_change_possibilities(
    egc,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    fragment_member_vector=None,
    max_fragment_num=None,
    exclude_equivalent=True,
    **other_kwargs,
):

    output = []
    cg = egc.chemgraph
    hatoms = cg.hatoms
    if bond_order_change == 0:
        return output
    for ha_id1, ha1 in enumerate(hatoms):
        nc1 = ha1.ncharge
        if not_protonated is not None:
            if nc1 in not_protonated:
                continue
        if bond_order_change > 0:
            if ha1.nhydrogens < bond_order_change:
                continue
        for ha_id2, ha2 in enumerate(hatoms[:ha_id1]):
            nc2 = ha2.ncharge
            if bond_order_change > 0:
                if ha2.nhydrogens < bond_order_change:
                    continue
            if not_protonated is not None:
                if nc2 in not_protonated:
                    continue
            bond_tuple = (ha_id1, ha_id2)
            if fragment_member_vector is not None:
                if fragment_member_vector[ha_id1] == fragment_member_vector[ha_id2]:
                    continue
            if connection_forbidden(nc1, nc2, forbidden_bonds):
                continue
            possible_bond_orders = cg.aa_all_bond_orders(*bond_tuple)
            max_bond_order = max(possible_bond_orders)
            if bond_order_change > 0:
                min_bond_order = min(possible_bond_orders)
                if min_bond_order + bond_order_change > max_bo(nc1, nc2):
                    continue
                if max_bond_order == min_bond_order:
                    possible_resonance_structures = [None]
                else:
                    possible_resonance_structures = [
                        cg.aa_all_bond_orders(*bond_tuple, unsorted=True).index(
                            min_bond_order
                        )
                    ]
            else:
                if max_bond_order < -bond_order_change:
                    continue
                # The results will be different if we apply the change to a resonance structure where the bond order equals -bond_order_change or not. The algorithm accounts for both options.
                unsorted_bond_orders = None
                possible_resonance_structures = []
                for pbo in possible_bond_orders:
                    if pbo == -bond_order_change:
                        if max_fragment_num is not None:
                            if cg.num_connected() == max_fragment_num:
                                if (
                                    cg.graph.edge_connectivity(
                                        source=ha_id1, target=ha_id2
                                    )
                                    == 1
                                ):
                                    continue
                    if pbo >= -bond_order_change:
                        if unsorted_bond_orders is None:
                            unsorted_bond_orders = cg.aa_all_bond_orders(
                                *bond_tuple, unsorted=True
                            )
                        possible_resonance_structures.append(
                            unsorted_bond_orders.index(pbo)
                        )
                        if pbo != -bond_order_change:
                            break

            if exclude_equivalent:
                if atom_pair_equivalent_to_list_member(egc, bond_tuple, output):
                    continue
                for poss_res_struct in possible_resonance_structures:
                    output.append((*bond_tuple, poss_res_struct))

    return output


def gen_val_change_pos_ncharges(possible_elements, not_protonated=None):
    output = []
    for pos_el in possible_elements:
        pos_ncharge = int_atom_checked(pos_el)

        cur_avail_val_list = avail_val_list(pos_ncharge)

        if isinstance(cur_avail_val_list, int):
            continue

        if not_protonated is not None:
            if pos_ncharge in not_protonated:
                continue
        output.append(pos_ncharge)
    return output


def valence_change_possibilities(
    egc,
    val_change_poss_ncharges=None,
    possible_elements=["C"],
    exclude_equivalent=True,
    not_protonated=None,
    **other_kwargs,
):

    if val_change_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_pos_ncharges(
            possible_elements, not_protonated=not_protonated
        )

    cg = egc.chemgraph
    cg.init_resonance_structures()

    output = {}

    for ha_id, ha in enumerate(cg.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, output):
                continue
        if ha.possible_valences is None:
            min_init_val = ha.valence
            min_res_struct = None
            max_init_val = ha.valence
            max_res_struct = None
        else:
            min_init_val = min(ha.possible_valences)
            min_res_struct = ha.possible_valences.index(min_init_val)
            max_init_val = max(ha.possible_valences)
            max_res_struct = ha.possible_valences.index(max_init_val)

        cur_val_list = ha.avail_val_list()
        available_valences = []
        for val in cur_val_list:
            if val > min_init_val:
                available_valences.append((val, min_res_struct))
            if val < max_init_val:
                if max_init_val - val <= ha.nhydrogens:
                    available_valences.append((val, max_res_struct))
        if len(available_valences) != 0:
            output[ha_id] = available_valences
    return output


def available_added_atom_bos(added_element, added_bond_orders, not_protonated=None):
    max_added_valence = default_valence(added_element)
    if not_protonated is not None:
        if int_atom_checked(added_element) in not_protonated:
            if max_added_valence in added_bond_orders:
                return [max_added_valence]
            else:
                return []
    output = []
    for abo in added_bond_orders:
        if abo <= max_added_valence:
            output.append(abo)
    return output


def gen_val_change_add_atom_pos_ncharges(
    possible_elements, chain_starting_element, forbidden_bonds=None
):
    val_change_pos_changes = gen_val_change_pos_ncharges(
        possible_elements, not_protonated=None
    )
    if forbidden_bonds is None:
        return val_change_pos_changes
    else:
        output = []
        for ncharge in val_change_pos_changes:
            if not connection_forbidden(
                ncharge, chain_starting_element, forbidden_bonds
            ):
                output.append(ncharge)
        return output


def valence_change_add_atoms_possibilities(
    egc,
    chain_starting_element,
    forbidden_bonds=None,
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    not_protonated=None,
    avail_added_bond_orders_val_change=None,
    val_change_add_atom_poss_ncharges=None,
    possible_elements=None,
    **other_kwargs,
):
    possibilities = {}
    if val_change_add_atom_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_add_atom_pos_ncharges(
            possible_elements, chain_starting_element, forbidden_bonds=forbidden_bonds
        )
    else:
        val_change_poss_ncharges = val_change_add_atom_poss_ncharges[
            chain_starting_element
        ]

    if len(val_change_poss_ncharges) == 0:
        return possibilities

    if avail_added_bond_orders_val_change is None:
        avail_bond_orders = available_added_atom_bos(
            chain_starting_element,
            added_bond_orders_val_change,
            not_protonated=not_protonated,
        )
    else:
        avail_bond_orders = avail_added_bond_orders_val_change[chain_starting_element]

    if nhatoms_range is not None:
        max_added_nhatoms = nhatoms_range[1] - egc.num_heavy_atoms()
        if max_added_nhatoms < 0:
            raise Exception

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue
        if ha.possible_valences is None:
            cur_valence = ha.valence
            valence_option = None
        else:
            cur_valence = min(ha.possible_valences)
            valence_option = ha.possible_valences.index(cur_valence)

        new_valence = next_valence(ha, valence_option_id=valence_option)
        if new_valence is None:
            continue
        val_diff = new_valence - cur_valence
        for added_bond_order in avail_bond_orders:
            if val_diff % added_bond_order != 0:
                continue
            added_nhatoms = val_diff // added_bond_order
            if (nhatoms_range is not None) and (added_nhatoms > max_added_nhatoms):
                continue
            if ha_id in possibilities:
                possibilities[ha_id].append(added_bond_order)
            else:
                possibilities[ha_id] = [added_bond_order]
    return possibilities


def valence_change_remove_atoms_possibilities(
    egc,
    removed_atom_type,
    possible_elements=["C"],
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    avail_added_bond_orders_val_change=None,
    val_change_add_atom_poss_ncharges=None,
    forbidden_bonds=None,
    not_protonated=None,
    default_valences=None,
    **other_kwargs,
):

    if nhatoms_range is not None:
        max_removed_nhatoms = egc.num_heavy_atoms() - nhatoms_range[0]
        if max_removed_nhatoms < 0:
            raise Exception()

    possibilities = {}
    if val_change_add_atom_poss_ncharges is None:
        val_change_poss_ncharges = gen_val_change_add_atom_pos_ncharges(
            possible_elements, removed_atom_type, forbidden_bonds=forbidden_bonds
        )
    else:
        val_change_poss_ncharges = val_change_add_atom_poss_ncharges[removed_atom_type]

    if len(val_change_poss_ncharges) == 0:
        return possibilities

    if avail_added_bond_orders_val_change is None:
        avail_bond_orders = available_added_atom_bos(
            removed_atom_type,
            added_bond_orders_val_change,
            not_protonated=not_protonated,
        )
    else:
        avail_bond_orders = avail_added_bond_orders_val_change[removed_atom_type]

    removed_atom_ncharge = int_atom_checked(removed_atom_type)

    if default_valences is None:
        default_removed_valence = default_valences[removed_atom_ncharge]
    else:
        default_removed_valence = default_valence(removed_atom_ncharge)

    cg = egc.chemgraph

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possibilities):
                continue
        if ha.possible_valences is None:
            valence_options = [None]
        else:
            valence_options = range(len(ha.possible_valences))

        found_options = []

        for val_opt in valence_options:
            if val_opt is None:
                cur_valence = ha.valence
            else:
                cur_valence = ha.possible_valences[val_opt]
            new_valence = next_valence(ha, int_step=-1, valence_option_id=val_opt)
            if new_valence is None:
                continue

            val_diff = cur_valence - new_valence

            for added_bond_order in avail_bond_orders:
                if added_bond_order in found_options:
                    continue
                if val_diff % added_bond_order != 0:
                    continue
                removed_nhatoms = val_diff // added_bond_order
                if (nhatoms_range is not None) and (
                    removed_nhatoms > max_removed_nhatoms
                ):
                    continue
                removed_hatoms = []
                for neigh in cg.neighbors(ha_id):
                    if cg.bond_order(ha_id, neigh) != added_bond_order:
                        continue
                    neigh_ha = cg.hatoms[neigh]
                    if neigh_ha.ncharge != removed_atom_ncharge:
                        continue
                    if cg.num_neighbors(neigh) != 1:
                        continue
                    if neigh_ha.valence != default_removed_valence:
                        continue
                    removed_hatoms.append(neigh)
                    if removed_nhatoms == len(removed_hatoms):
                        break
                if removed_nhatoms != len(removed_hatoms):
                    continue
                found_options.append(added_bond_order)
                poss_tuple = (tuple(removed_hatoms), val_opt)
                if ha_id in possibilities:
                    possibilities[ha_id].append(poss_tuple)
                else:
                    possibilities[ha_id] = [poss_tuple]
    return possibilities


# TODO add option for having opportunities as tuples vs dictionary? (PERHAPS NOT RELEVANT WITHOUT 4-order bonds)
def valence_bond_change_possibilities(
    egc,
    bond_order_change,
    forbidden_bonds=None,
    not_protonated=None,
    max_fragment_num=None,
    exclude_equivalent=True,
    **other_kwargs,
):

    cg = egc.chemgraph
    hatoms = cg.hatoms
    output = []
    if bond_order_change == 0:
        return output

    if bond_order_change < 0:
        disconnecting = {}

    for mod_val_ha_id, mod_val_ha in enumerate(hatoms):
        mod_val_nc = mod_val_ha.ncharge
        if not mod_val_ha.is_polyvalent():
            continue

        resonance_structure_region = cg.single_atom_resonance_structure(mod_val_ha_id)
        if resonance_structure_region is None:
            resonance_struct_ids = [None]
        else:
            res_struct_valence_vals = cg.resonance_structure_valence_vals[
                resonance_structure_region
            ]
            resonance_struct_ids = range(len(res_struct_valence_vals))
            res_struct_added_bos = cg.resonance_structure_orders[
                resonance_structure_region
            ]

        for other_ha_id, other_ha in enumerate(hatoms):
            if other_ha_id == mod_val_ha_id:
                continue

            other_nc = other_ha.ncharge

            if bond_order_change > 0:
                if connection_forbidden(mod_val_nc, other_nc, forbidden_bonds):
                    continue
                if hatoms[other_ha_id].nhydrogens < bond_order_change:
                    continue
            else:
                if not_protonated is not None:
                    if other_nc in not_protonated:
                        continue

            bond_tuple = (mod_val_ha_id, other_ha_id)

            if exclude_equivalent:
                if atom_pair_equivalent_to_list_member(egc, bond_tuple, output):
                    continue

            st = sorted_tuple(mod_val_ha_id, other_ha_id)

            for resonance_struct_id in resonance_struct_ids:
                if resonance_struct_id is not None:
                    cur_res_struct_added_bos = res_struct_added_bos[resonance_struct_id]
                if (resonance_struct_id is None) or (
                    mod_val_ha.possible_valences is None
                ):
                    valence_option_id = None
                    cur_mod_valence = mod_val_ha.valence
                else:
                    valence_option_id = res_struct_valence_vals[resonance_struct_id]
                    cur_mod_valence = mod_val_ha.possible_valences[valence_option_id]

                if (
                    next_valence(
                        mod_val_ha,
                        np.sign(bond_order_change),
                        valence_option_id=valence_option_id,
                    )
                    != cur_mod_valence + bond_order_change
                ):
                    continue

                cur_bo = cg.bond_order(mod_val_ha_id, other_ha_id)
                if (resonance_struct_id is not None) and (cur_bo != 0):
                    cur_bo = 1
                    if st in cur_res_struct_added_bos:
                        cur_bo += cur_res_struct_added_bos[st]

                if bond_order_change > 0:
                    if cur_bo + bond_order_change > max_bo(mod_val_nc, other_nc):
                        continue
                    if bond_tuple not in output:
                        output.append((*bond_tuple, resonance_struct_id))
                else:
                    if cur_bo < -bond_order_change:
                        continue
                    if cur_bo == -bond_order_change:
                        if (cg.num_connected() == max_fragment_num) and (
                            cg.graph.edge_connectivity(
                                source=mod_val_ha_id, target=other_ha_id
                            )
                            == 1
                        ):
                            continue
                        if bond_tuple in disconnecting:
                            if disconnecting[bond_tuple] == 1:
                                continue
                        if bond_tuple in disconnecting:
                            disconnecting[bond_tuple] = 2
                        else:
                            disconnecting[bond_tuple] = 1
                    else:
                        if bond_tuple in disconnecting:
                            if disconnecting[bond_tuple] == 0:
                                continue
                        if bond_tuple in disconnecting:
                            disconnecting[bond_tuple] = 2
                        else:
                            disconnecting[bond_tuple] = 0
                    if disconnecting[bond_tuple] != 2:
                        output.append((*bond_tuple, resonance_struct_id))

    return output


def val_min_checked_egc(cg):
    if cg.attempt_minimize_valences():
        return ExtGraphCompound(chemgraph=cg)
    else:
        return None


def add_heavy_atom_chain(
    egc, modified_atom_id, new_chain_atoms, chain_bond_orders=None
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.add_heavy_atom_chain(
        modified_atom_id, new_chain_atoms, chain_bond_orders=chain_bond_orders
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def replace_heavy_atom(
    egc,
    replaced_atom_id,
    inserted_atom,
    inserted_valence=None,
    resonance_structure_id=None,
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.replace_heavy_atom(
        replaced_atom_id,
        inserted_atom,
        inserted_valence=inserted_valence,
        resonance_structure_id=resonance_structure_id,
    )
    return val_min_checked_egc(new_chemgraph)


def remove_heavy_atom(egc, removed_atom_id, resonance_structure_id=None):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.remove_heavy_atom(
        removed_atom_id, resonance_structure_id=resonance_structure_id
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_bond_order(
    egc, atom_id1, atom_id2, bond_order_change, resonance_structure_id=0
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.change_bond_order(
        atom_id1,
        atom_id2,
        bond_order_change,
        resonance_structure_id=resonance_structure_id,
    )
    return val_min_checked_egc(new_chemgraph)


def change_valence(egc, modified_atom_id, new_valence, resonance_structure_id=None):
    new_chemgraph = deepcopy(egc.chemgraph)
    if resonance_structure_id is not None:
        resonance_structure_region = new_chemgraph.single_atom_resonance_structure(
            modified_atom_id
        )
        new_chemgraph.adjust_resonance_valences(
            resonance_structure_region, resonance_structure_id
        )
    new_chemgraph.change_valence(modified_atom_id, new_valence)
    return val_min_checked_egc(new_chemgraph)


def change_valence_add_atoms(egc, modified_atom_id, new_atom_element, new_bo):
    new_chemgraph = deepcopy(egc.chemgraph)

    mod_hatom = new_chemgraph.hatoms[modified_atom_id]

    if mod_hatom.possible_valences is not None:
        min_valence = min(mod_hatom.possible_valences)
        min_val_poss = mod_hatom.possible_valences.index(min_valence)
        new_chemgraph.adjust_resonance_valences_atom(
            modified_atom_id, valence_option_id=min_val_poss
        )

    new_atom_charge = int_atom_checked(new_atom_element)

    new_mod_valence_val = next_valence(mod_hatom)
    val_diff = new_mod_valence_val - mod_hatom.valence
    if val_diff % new_bo != 0:
        raise Exception()
    num_added = val_diff // new_bo
    new_chemgraph.change_valence(modified_atom_id, new_mod_valence_val)
    for _ in range(num_added):
        new_chemgraph.add_heavy_atom_chain(
            modified_atom_id, [new_atom_charge], chain_bond_orders=[new_bo]
        )
    return val_min_checked_egc(new_chemgraph)


def change_valence_remove_atoms(
    egc, modified_atom_id, removed_neighbors, resonance_structure_id=None
):
    new_chemgraph = deepcopy(egc.chemgraph)

    new_chemgraph.adjust_resonance_valences_atom(
        modified_atom_id, resonance_structure_id=resonance_structure_id
    )

    mod_hatom = new_chemgraph.hatoms[modified_atom_id]
    new_mod_valence_val = next_valence(mod_hatom, -1)
    val_diff = mod_hatom.valence - new_mod_valence_val

    id_shift = 0
    running_bond = None
    for neigh in removed_neighbors:
        if neigh < modified_atom_id:
            id_shift -= 1
        cur_bond = new_chemgraph.bond_order(neigh, modified_atom_id)
        if running_bond is None:
            running_bond = cur_bond
            if running_bond == 0:
                raise Exception()
        else:
            if cur_bond != running_bond:
                raise Exception()
        val_diff -= cur_bond
        if val_diff == 0:
            break
    if val_diff != 0:
        raise Exception()
    new_chemgraph.remove_heavy_atoms(removed_neighbors)
    new_modified_atom_id = modified_atom_id + id_shift
    new_chemgraph.change_valence(new_modified_atom_id, new_mod_valence_val)
    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_bond_order_valence(
    egc,
    val_changed_atom_id,
    other_atom_id,
    bond_order_change,
    resonance_structure_id=None,
):

    new_chemgraph = deepcopy(egc.chemgraph)

    new_chemgraph.adjust_resonance_valences_atom(
        val_changed_atom_id, resonance_structure_id=resonance_structure_id
    )

    new_valence = new_chemgraph.hatoms[val_changed_atom_id].valence + bond_order_change

    if bond_order_change > 0:
        new_chemgraph.change_valence(val_changed_atom_id, new_valence)

    new_chemgraph.change_bond_order(
        val_changed_atom_id, other_atom_id, bond_order_change
    )

    if bond_order_change < 0:
        new_chemgraph.change_valence(val_changed_atom_id, new_valence)

    if not new_chemgraph.hatoms[val_changed_atom_id].valence_reasonable():
        raise Exception()

    return val_min_checked_egc(new_chemgraph)


# Procedures for genetic algorithms.
class FragmentPair:
    def __init__(self, cg: ChemGraph, membership_vector: list or np.array):
        """
        Pair of fragments formed from one molecule split around the membership vector.
        cg : ChemGraph - the split molecule
        membership_vector : list or NumPy array - list of integer values indicating which fragment a given HeavyAtom belongs to.
        """
        self.chemgraph = cg

        self.chemgraph.init_resonance_structures()

        self.membership_vector = membership_vector
        # Two list of vertices corresponding to the two grahments.
        self.sorted_vertices = sorted_by_membership(self.membership_vector)
        self.affected_resonance_structures = []

        # Find resonance structures affected by the bond split.
        resonance_structure_orders_iterators = []

        resonance_structure_affected_bonds = {}
        saved_all_bond_orders = {}

        default_bond_order_dict = {}

        for i in self.sorted_vertices[0]:
            for neigh in self.chemgraph.neighbors(i):
                if self.membership_vector[neigh] != 0:
                    bond_tuple = (i, neigh)
                    bond_stuple = sorted_tuple(*bond_tuple)
                    if bond_stuple in self.chemgraph.resonance_structure_map:
                        rsr_id = self.chemgraph.resonance_structure_map[bond_stuple]
                        if rsr_id not in saved_all_bond_orders:
                            saved_all_bond_orders[rsr_id] = {}
                            self.affected_resonance_structures.append(rsr_id)
                        resonance_structure_orders_iterators.append(
                            range(
                                len(self.chemgraph.resonance_structure_orders[rsr_id])
                            )
                        )
                        if rsr_id not in resonance_structure_affected_bonds:
                            resonance_structure_affected_bonds[rsr_id] = []
                        resonance_structure_affected_bonds[rsr_id].append(bond_tuple)
                        saved_all_bond_orders[rsr_id][
                            bond_tuple
                        ] = self.chemgraph.aa_all_bond_orders(i, neigh, unsorted=True)

                    else:
                        cur_bo = self.chemgraph.bond_orders[bond_stuple]
                        if cur_bo in default_bond_order_dict:
                            default_bond_order_dict[cur_bo].append(bond_tuple)
                        else:
                            default_bond_order_dict[cur_bo] = [bond_tuple]

        if len(resonance_structure_orders_iterators) == 0:
            self.affected_status = [{"bonds": default_bond_order_dict, "valences": {}}]
        else:
            self.affected_status = []
            for resonance_structure_orders_ids in itertools.product(
                *resonance_structure_orders_iterators
            ):
                new_status = {
                    "bonds": deepcopy(default_bond_order_dict),
                    "valences": {},
                }
                for res_reg_id, rso_id in zip(
                    self.affected_resonance_structures, resonance_structure_orders_ids
                ):
                    ha_ids = self.chemgraph.resonance_structure_inverse_map[res_reg_id]
                    val_pos = self.chemgraph.resonance_structure_valence_vals[
                        res_reg_id
                    ][rso_id]
                    for ha_id in ha_ids:
                        poss_valences = self.chemgraph.hatoms[ha_id].possible_valences
                        if poss_valences is not None:
                            new_status["valences"][ha_id] = poss_valences[val_pos]
                    for btuple, bos in saved_all_bond_orders[res_reg_id].items():
                        cur_bo = bos[rso_id]
                        if cur_bo in new_status["bonds"]:
                            new_status["bonds"][cur_bo].append(btuple)
                        else:
                            new_status["bonds"][cur_bo] = [btuple]

                if new_status not in self.affected_status:
                    self.affected_status.append(new_status)

    def adjusted_ha_valences(self, status_id, membership_id):
        vals = self.affected_status[status_id]["valences"]
        output = []
        for i in self.sorted_vertices[membership_id]:
            if i in vals:
                output.append(vals[i])
            else:
                output.append(self.chemgraph.hatoms[i].valence)
        return output

    def cross_couple(
        self,
        other_fp,
        switched_bond_tuples_self: int,
        switched_bond_tuples_other: int,
        affected_status_id_self: int,
        affected_status_id_other: int,
        forbidden_bonds=None,
    ):
        """
        Couple to another fragment.
        switched_bond_tuples_self, switched_bond_tuples_other - bonds corresponding to which heavy atom index tuples are switched between different fragments.
        affected_bonds_id_self, affected_bonds_id_other - which sets of affected bond orders the fragments are initialized with.
        """

        remainder_subgraph = self.chemgraph.graph.subgraph(self.sorted_vertices[0])
        other_frag_subgraph = other_fp.chemgraph.graph.subgraph(
            other_fp.sorted_vertices[1]
        )

        nhatoms1 = len(self.sorted_vertices[0])
        nhatoms2 = len(other_fp.sorted_vertices[1])

        new_membership_vector = np.append(
            np.zeros((nhatoms1,), dtype=int), np.ones((nhatoms2,), dtype=int)
        )

        new_hatoms = [
            deepcopy(self.chemgraph.hatoms[ha_id]) for ha_id in self.sorted_vertices[0]
        ] + [
            deepcopy(other_fp.chemgraph.hatoms[ha_id])
            for ha_id in other_fp.sorted_vertices[1]
        ]

        new_hatoms_old_valences = self.adjusted_ha_valences(
            affected_status_id_self, 0
        ) + other_fp.adjusted_ha_valences(affected_status_id_other, 1)

        new_graph = disjoint_union([remainder_subgraph, other_frag_subgraph])

        created_bonds = []

        for btuple1, btuple2 in zip(
            switched_bond_tuples_self, switched_bond_tuples_other
        ):
            internal_id1 = self.sorted_vertices[0].index(btuple1[0])
            internal_id2 = other_fp.sorted_vertices[1].index(btuple2[1]) + nhatoms1

            new_bond_tuple = (internal_id1, internal_id2)
            if new_bond_tuple in created_bonds:
                return None, None  # from detailed balance concerns
            else:
                created_bonds.append(new_bond_tuple)

            new_graph.add_edge(*new_bond_tuple)
            if forbidden_bonds is not None:
                if connection_forbidden(
                    *[
                        new_hatoms[internal_id].ncharge
                        for internal_id in new_bond_tuple
                    ],
                    forbidden_bonds,
                ):
                    return None, None

        new_ChemGraph = ChemGraph(hatoms=new_hatoms, graph=new_graph)

        # Lastly, check that re-initialization does not decrease the bond order.
        if new_ChemGraph.valence_config_valid(new_hatoms_old_valences):
            return new_ChemGraph, new_membership_vector
        else:
            return None, None


def bond_dicts_match(frag_bond_dict1: dict, frag_bond_dict2: dict):
    """
    Check that two results of current_affected_bonds call in FragmentPair correspond to fragments that can be cross-coupled.
    """
    if len(frag_bond_dict1) != len(frag_bond_dict2):
        return False
    for key1, positions1 in frag_bond_dict1.items():
        if key1 not in frag_bond_dict2:
            return False
        if len(positions1) != len(frag_bond_dict2[key1]):
            return False
    return True


def matching_dict_tuples(fp1: FragmentPair, fp2: FragmentPair):
    """
    Check which valence configurations of FragmentPair instances can be used for cross-coupling. List of tuples of their indices is returned as output.
    """
    output = []
    for as1, affected_status1 in enumerate(fp1.affected_status):
        for as2, affected_status2 in enumerate(fp2.affected_status):
            if bond_dicts_match(affected_status1["bonds"], affected_status2["bonds"]):
                output.append((as1, as2))
    return output


def random_connect_fragments(
    fp_pair: tuple, connection_tuple: tuple, forbidden_bonds: list or None = None
):
    """
    Randonly reconnect two FragmentPair objects into new ChemGraph objects.
    """

    switched_bond_tuples = [[], []]

    bond_dicts = [
        fp.affected_status[affected_bonds_id]["bonds"]
        for fp, affected_bonds_id in zip(fp_pair, connection_tuple)
    ]

    for key, l0 in bond_dicts[0].items():
        l1 = bond_dicts[1][key]
        random.shuffle(l1)
        switched_bond_tuples[0] += l0
        switched_bond_tuples[1] += l1

    new_mols = []
    new_membership_vectors = []
    for self_id in [0, 1]:
        other_id = 1 - self_id
        cur_new_mol, cur_new_membership_vector = fp_pair[self_id].cross_couple(
            fp_pair[other_id],
            switched_bond_tuples[self_id],
            switched_bond_tuples[other_id],
            connection_tuple[self_id],
            connection_tuple[other_id],
            forbidden_bonds=forbidden_bonds,
        )
        new_mols.append(cur_new_mol)
        new_membership_vectors.append(cur_new_membership_vector)

    return new_mols, new_membership_vectors


def possible_fragment_sizes(
    cg, fragment_ratio_range=[0.0, 0.5], fragment_size_range=None
):
    """
    Possible fragment sizes for a ChemGraph objects satisfying enforced constraints.
    """
    if fragment_ratio_range is None:
        bounds = deepcopy(fragment_size_range)
    else:
        bounds = [int(r * cg.nhatoms()) for r in fragment_ratio_range]
    if bounds[1] >= cg.nhatoms():
        bounds[1] = cg.nhatoms - 1
    for j in range(2):
        if bounds[j] == 0:
            bounds[j] = 1
    return range(bounds[0], bounds[1] + 1)


def possible_pair_fragment_sizes(
    cg_pair, nhatoms_range=None, **possible_fragment_sizes_kwargs
):
    """
    Possible sizes of fragments for ChemGraph objects in cg_pair that lead to cross-coupled molecules satisfying size constraints enforced by nhatoms_range and other keyword arguments.
    """
    possible_fragment_sizes_iterators = []
    for cg in cg_pair:
        possible_fragment_sizes_iterators.append(
            possible_fragment_sizes(cg, **possible_fragment_sizes_kwargs)
        )
    output = []
    for size_tuple in itertools.product(*possible_fragment_sizes_iterators):
        if nhatoms_range is not None:
            unfit = False
            for cg, ordered_size_tuple in zip(
                cg_pair, itertools.permutations(size_tuple)
            ):
                new_mol_size = (
                    cg.nhatoms() + ordered_size_tuple[1] - ordered_size_tuple[0]
                )
                if (new_mol_size < nhatoms_range[0]) or (
                    new_mol_size > nhatoms_range[1]
                ):
                    unfit = True
                    break
            if unfit:
                continue
        output.append(size_tuple)
    return output


def randomized_split_membership_vector(cg, origin_choices, fragment_size):
    """
    Randomly create a membership vector that can be used to split ChemGraph object into a FragmentPair object.
    """
    membership_vector = np.zeros(cg.nhatoms(), dtype=int)

    origin_choices = cg.unrepeated_atom_list()

    start_id = random.choice(origin_choices)

    prev_to_add = [start_id]
    remaining_atoms = fragment_size
    frag_id = 1
    while (len(prev_to_add) != 0) and (remaining_atoms != 0):
        for added_id in prev_to_add:
            membership_vector[added_id] = frag_id
        remaining_atoms -= len(prev_to_add)
        if remaining_atoms == 0:
            break
        new_to_add = []
        for added_id in prev_to_add:
            for neigh in cg.neighbors(added_id):
                if (membership_vector[neigh] == 0) and (neigh not in new_to_add):
                    new_to_add.append(neigh)
        if len(new_to_add) > remaining_atoms:
            random.shuffle(new_to_add)
            del new_to_add[: len(new_to_add) - remaining_atoms]
        prev_to_add = new_to_add

    return membership_vector


def randomized_cross_coupling(
    cg_pair: list,
    cross_coupling_fragment_ratio_range: list or None = [0.0, 0.5],
    cross_coupling_fragment_size_range: list or None = None,
    forbidden_bonds: list or None = None,
    nhatoms_range: list or None = None,
    visited_tp_list: list or None = None,
    **dummy_kwargs,
):
    """
    Break two ChemGraph objects into two FragmentPair objects; the fragments are then re-coupled into two new ChemGraph objects.
    """

    ppfs_kwargs = {
        "nhatoms_range": nhatoms_range,
        "fragment_ratio_range": cross_coupling_fragment_ratio_range,
        "fragment_size_range": cross_coupling_fragment_size_range,
    }

    pair_fragment_sizes = possible_pair_fragment_sizes(cg_pair, **ppfs_kwargs)

    if len(pair_fragment_sizes) == 0:
        return None, None

    # tot_choice_prob_ratio is the ratio of probability of the trial move divided by probability of the inverse move.
    tot_choice_prob_ratio = 1.0 / float(len(pair_fragment_sizes))

    final_pair_fragment_sizes = random.choice(pair_fragment_sizes)

    membership_vectors = []
    for cg, fragment_size in zip(cg_pair, final_pair_fragment_sizes):
        origin_choices = cg.unrepeated_atom_list()
        tot_choice_prob_ratio /= len(origin_choices)
        membership_vectors.append(
            randomized_split_membership_vector(cg, origin_choices, fragment_size)
        )

    fragment_pairs = [
        FragmentPair(cg, membership_vector)
        for cg, membership_vector in zip(cg_pair, membership_vectors)
    ]

    mdtuples = matching_dict_tuples(*fragment_pairs)

    if len(mdtuples) == 0:
        return None, None

    tot_choice_prob_ratio /= len(mdtuples)

    final_md_tuple = random.choice(mdtuples)

    new_cg_pair, new_membership_vectors = random_connect_fragments(
        fragment_pairs, final_md_tuple, forbidden_bonds=forbidden_bonds
    )

    if (new_cg_pair[0] is None) or (new_cg_pair[1] is None):
        return None, None

    if visited_tp_list is not None:
        for new_cg_id, new_cg in enumerate(new_cg_pair):
            if new_cg in visited_tp_list:
                visited_tp_list[
                    visited_tp_list.index(new_cg)
                ].chemgraph().copy_extra_data_to(new_cg_pair[new_cg_id])

    # Account for probability of choosing the correct fragment sizes to do the inverse move.
    tot_choice_prob_ratio *= len(
        possible_pair_fragment_sizes(new_cg_pair, **ppfs_kwargs)
    )
    # Account for probability of choosing the necessary resonance structure.
    backwards_fragment_pairs = [
        FragmentPair(new_cg, new_membership_vector)
        for new_cg, new_membership_vector in zip(new_cg_pair, new_membership_vectors)
    ]
    backwards_mdtuples = matching_dict_tuples(*backwards_fragment_pairs)

    tot_choice_prob_ratio *= len(backwards_mdtuples)

    for new_cg in new_cg_pair:
        tot_choice_prob_ratio *= len(new_cg.unrepeated_atom_list())

    return new_cg_pair, np.log(tot_choice_prob_ratio)


def no_forbidden_bonds(egc: ExtGraphCompound, forbidden_bonds: None or list = None):
    """
    Check that an ExtGraphCompound object has no covalent bonds whose nuclear charge tuple is inside forbidden_bonds.
    egc : checked ExtGraphCompound object
    forbidden_bonds : list of sorted nuclear charge tuples.
    """
    if forbidden_bonds is not None:
        cg = egc.chemgraph
        hatoms = cg.hatoms
        for bond_tuple in cg.bond_orders.keys():
            if connection_forbidden(
                hatoms[bond_tuple[0]].ncharge,
                hatoms[bond_tuple[1]].ncharge,
                forbidden_bonds=forbidden_bonds,
            ):
                return False
    return True


def egc_valid_wrt_change_params(
    egc,
    nhatoms_range=None,
    forbidden_bonds=None,
    possible_elements=None,
    not_protonated=None,
    **other_kwargs,
):
    """
    Check that an ExtGraphCompound object is a member of chemical subspace spanned by change params used throughout chemxpl.modify module.
    egc : ExtGraphCompound object
    nhatoms_range : range of possible numbers of heavy atoms
    forbidden_bonds : ordered tuples of nuclear charges corresponding to elements that are forbidden to have bonds.
    """
    if not no_forbidden_bonds(egc, forbidden_bonds=forbidden_bonds):
        return False
    if not_protonated is not None:
        for ha in egc.chemgraph.hatoms:
            if (ha.ncharge in not_protonated) and (ha.nhydrogens != 0):
                return False
    if nhatoms_range is not None:
        nhas = egc.chemgraph.nhatoms()
        if (nhas < nhatoms_range[0]) or (nhas > nhatoms_range[1]):
            return False
    if possible_elements is not None:
        possible_elements_nc = [
            NUCLEAR_CHARGE[canonical_atomtype(pe)] for pe in possible_elements
        ]
        for ha in egc.chemgraph.hatoms:
            if ha.ncharge not in possible_elements_nc:
                return False
    return True
