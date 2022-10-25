# TODO Perhaps add support for atoms with higher valences being added directly?
# TODO Ensure resonance structure invariance for genetic algorithms.

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
    non_default_valences,
)
from copy import deepcopy
import random, itertools
from igraph.operators import disjoint_union
from ..utils import canonical_atomtype
from ..data import NUCLEAR_CHARGE


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


# Some modification functions for EGC objects.
# TO-DO: does it make sense to get rid of them, leaving only ChemGraph functions?
def atom_replacement_possibilities(
    egc,
    inserted_atom,
    inserted_valence=None,
    replaced_atom=None,
    forbidden_bonds=None,
    exclude_equivalent=True,
    not_protonated=None,
    **other_kwargs,
):
    possible_ids = []
    inserted_iac = int_atom_checked(inserted_atom)
    if replaced_atom is not None:
        replaced_iac = int_atom_checked(replaced_atom)
    if inserted_valence is None:
        inserted_valence = default_valence(inserted_iac)
    if not_protonated is not None:
        cant_be_protonated = inserted_iac in not_protonated
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if replaced_atom is not None:
            if ha.ncharge != replaced_iac:
                continue
        if inserted_iac == ha.ncharge:
            continue
        if forbidden_bonds is not None:
            cont = False
            for neigh in egc.chemgraph.graph.neighbors(ha_id):
                if connection_forbidden(
                    egc.nuclear_charges[neigh], inserted_atom, forbidden_bonds
                ):
                    cont = True
                    break
            if cont:
                continue
        if default_valence(ha.ncharge) != ha.valence:
            continue
        val_diff = ha.valence - inserted_valence
        if val_diff <= ha.nhydrogens:
            if not_protonated is not None:
                if cant_be_protonated and (val_diff != ha.nhydrogens):
                    continue
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            possible_ids.append(ha_id)
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
        if (
            (ha.ncharge != deleted_iac)
            or (deleted_default_valence != ha.valence)
            or (ha.nhydrogens not in possible_hnums)
        ):
            continue
        neighs = cg.neighbors(ha_id)
        if len(neighs) != 1:
            continue
        if not_protonated is not None:
            if hatoms[neighs[0]].ncharge in not_protonated:
                continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        possible_ids.append(ha_id)
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

    natoms = egc.num_heavy_atoms()
    output = []
    if bond_order_change != 0:
        for i in range(natoms):
            nc1 = egc.nuclear_charges[i]
            if not_protonated is not None:
                if nc1 in not_protonated:
                    continue
            for j in range(i):
                nc2 = egc.nuclear_charges[j]
                if not_protonated is not None:
                    if nc2 in not_protonated:
                        continue
                bond_tuple = (j, i)
                if fragment_member_vector is not None:
                    if fragment_member_vector[i] == fragment_member_vector[j]:
                        continue
                if connection_forbidden(nc1, egc.nuclear_charges[j], forbidden_bonds):
                    continue
                possible_bond_orders = egc.chemgraph.aa_all_bond_orders(*bond_tuple)
                max_bond_order = max(possible_bond_orders)
                if bond_order_change > 0:
                    suitable = True
                    for q in bond_tuple:
                        suitable = suitable and (
                            egc.chemgraph.hatoms[q].nhydrogens >= bond_order_change
                        )
                    if suitable:
                        min_bond_order = min(possible_bond_orders)
                        suitable = min(
                            possible_bond_orders
                        ) + bond_order_change <= max_bo(nc1, nc2)
                    if suitable:
                        if max_bond_order == min_bond_order:
                            possible_resonance_structures = [0]
                        else:
                            possible_resonance_structures = [
                                egc.chemgraph.aa_all_bond_orders(
                                    *bond_tuple, unsorted=True
                                ).index(min_bond_order)
                            ]
                else:
                    suitable = max_bond_order >= -bond_order_change
                    if suitable and (max_fragment_num is not None):
                        if max_bond_order == -bond_order_change:
                            if egc.chemgraph.num_connected() == max_fragment_num:
                                suitable = (
                                    egc.chemgraph.graph.edge_connectivity(
                                        source=i, target=j
                                    )
                                    != 1
                                )
                    if suitable:
                        # If there is a resonance structure for which decreasing bond order breaks the bond then the result of
                        # the operation will depend on resonance structure chosen.
                        if (
                            (-bond_order_change in possible_bond_orders)
                            and (-bond_order_change != max_bond_order)
                            and (len(possible_bond_orders) != 1)
                        ):
                            unsorted_possible_bond_orders = (
                                egc.chemgraph.aa_all_bond_orders(
                                    *bond_tuple, unsorted=True
                                )
                            )
                            bond_break_index = unsorted_possible_bond_orders.index(
                                -bond_order_change
                            )
                            max_bond_index = unsorted_possible_bond_orders.index(
                                max_bond_order
                            )
                            possible_resonance_structures = [bond_break_index]
                            if max_bond_index != bond_break_index:
                                possible_resonance_structures.append(max_bond_index)
                        else:
                            possible_resonance_structures = [0]
                if suitable:
                    if exclude_equivalent:
                        if atom_pair_equivalent_to_list_member(egc, bond_tuple, output):
                            continue
                    for poss_res_struct in possible_resonance_structures:
                        output.append((*bond_tuple, poss_res_struct))

    return output


# TODO: initialize valence_change_possible_elements in random_walk?
def gen_val_change_pos_ncharges(possible_elements=["C"], not_protonated=None):
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
            possible_elements=possible_elements, not_protonated=not_protonated
        )

    output = {}

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ha.ncharge not in val_change_poss_ncharges:
            continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, output):
                continue
        cur_valence = ha.valence
        cur_val_list = ha.avail_val_list()
        cur_val_id = cur_val_list.index(cur_valence)
        available_valences = list(cur_val_list[cur_val_id + 1 :])
        for potential_valence in cur_val_list[:cur_val_id]:
            if cur_valence - potential_valence <= ha.nhydrogens:
                available_valences.append(potential_valence)
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


def valence_change_add_atom_possibilities(
    egc,
    chain_starting_element,
    forbidden_bonds=None,
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    not_protonated=None,
    **other_kwargs,
):

    if nhatoms_range is not None:
        max_added_nhatoms = nhatoms_range[-1] - egc.num_heavy_atoms()
        if max_added_nhatoms < 0:
            raise Exception

    avail_bond_orders = available_added_atom_bos(
        chain_starting_element,
        added_bond_orders_val_change,
        not_protonated=not_protonated,
    )

    output = []
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        valence_list = ha.avail_val_list()
        if isinstance(valence_list, tuple):
            if connection_forbidden(
                ha.ncharge, chain_starting_element, forbidden_bonds
            ):
                continue
            cur_valence = ha.valence
            if cur_valence == valence_list[-1]:
                continue
            val_diff = valence_list[valence_list.index(cur_valence) + 1] - cur_valence
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, output):
                    continue
            for added_bond_order in avail_bond_orders:
                if val_diff % added_bond_order != 0:
                    continue
                added_nhatoms = val_diff // added_bond_order
                if (nhatoms_range is not None) and (added_nhatoms > max_added_nhatoms):
                    continue
                output.append((ha_id, added_bond_order))
    return output


def valence_change_remove_atom_possibilities(
    egc,
    removed_atom_type,
    exclude_equivalent=True,
    nhatoms_range=None,
    added_bond_orders_val_change=[1, 2],
    **other_kwargs,
):

    if nhatoms_range is not None:
        max_removed_nhatoms = egc.num_heavy_atoms() - nhatoms_range[0]
        if max_removed_nhatoms < 0:
            raise Exception()

    removed_atom_ncharge = int_atom_checked(removed_atom_type)

    max_removed_valence = default_valence(removed_atom_type)
    output = {}
    if exclude_equivalent:
        found_atoms = []

    cg = egc.chemgraph

    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        valence_list = ha.avail_val_list()
        if isinstance(valence_list, tuple):
            cur_valence = ha.valence
            if cur_valence == valence_list[0]:
                continue
            val_diff = cur_valence - valence_list[valence_list.index(cur_valence) - 1]
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, found_atoms):
                    continue
            for added_bond_order in added_bond_orders_val_change:
                if added_bond_order > max_removed_valence:
                    continue
                if val_diff % added_bond_order != 0:
                    continue
                removed_nhatoms = val_diff // added_bond_order
                if (nhatoms_range is not None) and (
                    removed_nhatoms > max_removed_nhatoms
                ):
                    continue
                for neigh in cg.neighbors(ha_id):
                    if (
                        cg.bond_order(ha_id, neigh) != added_bond_order
                    ):  # TODO better way to rewrite?
                        continue
                    if cg.hatoms[neigh].ncharge != removed_atom_ncharge:
                        continue
                    if (
                        len(cg.neighbors(neigh)) != 1
                    ):  # TODO make nneigh a thing? Check igraph routines?
                        continue
                    removed_atoms -= 1
                    if removed_atoms == 0:
                        break
                if removed_nhatoms != 0:
                    continue
                if ha_id not in output:
                    output[ha_id] = []
                    if exclude_equivalent:
                        found_atoms.append(ha_id)
                output[ha_id].append(added_bond_order)
    return output


def add_heavy_atom_chain(
    egc, modified_atom_id, new_chain_atoms, chain_bond_orders=None
):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.add_heavy_atom_chain(
        modified_atom_id, new_chain_atoms, chain_bond_orders=chain_bond_orders
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def replace_heavy_atom(egc, replaced_atom_id, inserted_atom, inserted_valence=None):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.replace_heavy_atom(
        replaced_atom_id, inserted_atom, inserted_valence=inserted_valence
    )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def remove_heavy_atom(egc, removed_atom_id):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.remove_heavy_atom(removed_atom_id)
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

    if new_chemgraph.non_default_valence_present():
        old_non_default_valences = new_chemgraph.non_default_valences()
        new_chemgraph.reassign_nonsigma_bonds()
        if old_non_default_valences != new_chemgraph.non_default_valences():
            return None

    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_valence(egc, modified_atom_id, new_valence):
    new_chemgraph = deepcopy(egc.chemgraph)
    new_chemgraph.change_valence(modified_atom_id, new_valence)
    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_valence_add_atom(egc, modified_atom_id, new_atom_charge, new_bo):
    new_chemgraph = deepcopy(egc.chempgraph)
    mod_hatom = new_chemgraph.hatoms[modified_atom_id]
    cur_mod_valence_val = mod_hatom.valence
    cur_mod_valence_val_id = mod_hatom.valence_val_id()
    new_mod_valence_val = mod_hatom.avail_val_list()[cur_mod_valence_val_id + 1]
    val_diff = new_mod_valence_val - cur_mod_valence_val
    if val_diff % new_bo != 0:
        raise Exception()
    num_removed = val_diff // new_bo
    new_chemgraph.change_valence(modified_atom_id, new_mod_valence_val)
    for _ in range(num_removed):
        new_chemgraph.add_heavy_atom_chain(
            modified_atom_id, [new_atom_charge], chain_bond_orders=[new_bo]
        )
    return ExtGraphCompound(chemgraph=new_chemgraph)


def change_valence_remove_atom(egc, modified_atom_id, removed_atom_charge, removed_bo):
    new_chemgraph = deepcopy(egc.chempgraph)
    mod_hatom = new_chemgraph.hatoms[modified_atom_id]
    cur_mod_valence_val_id = mod_hatom.valence_val_id()
    cur_mod_valence = mod_hatom.valence
    new_mod_valence_val = mod_hatom.avail_val_list()[cur_mod_valence_val_id - 1]
    val_diff = cur_mod_valence - new_mod_valence_val
    if val_diff % removed_bo != 0:
        raise Exception()
    num_removed = val_diff // removed_bo
    deleted_atoms = []
    neighs = new_chemgraph.neighbors(modified_atom_id)
    for n in neighs:
        neigh_hatom = new_chemgraph.hatoms[n]
        # Check the charge is correct
        if neigh_hatom.ncharge != removed_atom_charge:
            continue
        if neigh_hatom.valence_val_id() != 0:
            continue
        if len(new_chemgraph.neighbors(n)) != 1:
            continue
        deleted_atoms.append(n)
        old_bo = new_chemgraph.bond_order(modified_atom_id, n)
        new_chemgraph.change_bond_order(modified_atom_id, n, -old_bo)
        num_removed -= 1
        if num_removed == 0:
            break
    if num_removed != 0:
        raise Exception()
    new_chemgraph.change_valence(modified_atom_id, new_mod_valence_val)
    new_chemgraph.remove_heavy_atoms(deleted_atoms)
    return ExtGraphCompound(chemgraph=new_chemgraph)


# Procedures for genetic algorithms.

# Pair of fragments formed from one molecule split around the membership vector.
class FragmentPair:
    def __init__(self, cg, membership_vector, copied=False):
        cg.init_resonance_structures()
        if copied:
            self.chemgraph = cg
        else:
            self.chemgraph = deepcopy(cg)

        self.membership_vector = membership_vector
        self.sorted_vertices = sorted_by_membership(self.membership_vector)
        self.affected_resonance_structures = []

        resonance_structure_orders_iterators = []

        for rsr_id, rsr_nodelist in enumerate(
            self.chemgraph.resonance_structure_inverse_map
        ):
            for i in rsr_nodelist[:-1]:
                if membership_vector[i] != membership_vector[-1]:
                    self.affected_resonance_structures.append(rsr_id)
                    resonance_structure_orders_iterators.append(
                        range(len(self.chemgraph.resonance_structure_orders[rsr_id]))
                    )
                    break

        if len(resonance_structure_orders_iterators) == 0:
            self.affected_bonds = [self.current_affected_bonds()]
            self.resonance_structure_adjustments = None
        else:
            self.affected_bonds = []
            self.resonance_structure_adjustments = []
            for resonance_structure_orders_ids in itertools.product(
                *resonance_structure_orders_iterators
            ):
                for resonance_structure_region_id, resonance_structure_orders_id in zip(
                    self.affected_resonance_structures, resonance_structure_orders_ids
                ):
                    self.chemgraph.adjust_resonance_valences(
                        resonance_structure_region_id, resonance_structure_orders_id
                    )
                cur_affected_bonds = self.current_affected_bonds()
                if cur_affected_bonds not in self.affected_bonds:
                    self.affected_bonds.append(cur_affected_bonds)
                    self.resonance_structure_adjustments.append(
                        resonance_structure_orders_ids
                    )

    def current_affected_bonds(self):
        output = {}
        for i in self.sorted_vertices[0]:
            for neigh in self.chemgraph.neighbors(i):
                if self.membership_vector[neigh] == 1:
                    btuple = (i, neigh)
                    bo = self.chemgraph.bond_order(*btuple)
                    if bo in output:
                        output[bo].append(btuple)
                    else:
                        output[bo] = [btuple]
        return output

    def cross_couple(
        self,
        other_fp,
        switched_bond_tuples_self,
        switched_bond_tuples_other,
        forbidden_bonds=None,
    ):
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

        init_non_default_valences = non_default_valences(new_hatoms)
        new_ChemGraph = ChemGraph(hatoms=new_hatoms, graph=new_graph)
        # Lastly, check that re-initialization does not decrease the bond order.
        new_non_default_valences = new_ChemGraph.non_default_valences()
        if new_non_default_valences == init_non_default_valences:
            return new_ChemGraph, new_membership_vector
        else:
            return None, None


def bond_dicts_match(frag_bond_dict1, frag_bond_dict2):
    if len(frag_bond_dict1) != len(frag_bond_dict2):
        return False
    for key1, positions1 in frag_bond_dict1.items():
        if key1 not in frag_bond_dict2:
            return False
        if len(positions1) != len(frag_bond_dict2[key1]):
            return False
    return True


def matching_dict_tuples(fp1, fp2):
    output = []
    for bdid1, bond_dict1 in enumerate(fp1.affected_bonds):
        for bdid2, bond_dict2 in enumerate(fp2.affected_bonds):
            if bond_dicts_match(bond_dict1, bond_dict2):
                output.append((bdid1, bdid2))
    return output


def random_connect_fragments(fp1, fp2, connection_tuple, forbidden_bonds=None):
    switched_bond_tuples1 = []
    switched_bond_tuples2 = []

    bond_dict1 = fp1.affected_bonds[connection_tuple[0]]
    bond_dict2 = fp2.affected_bonds[connection_tuple[1]]

    for key, l1 in bond_dict1.items():
        l2 = bond_dict2[key]
        random.shuffle(l1)
        switched_bond_tuples1 += l1
        switched_bond_tuples2 += l2

    new_mol1, new_membership_vector1 = fp1.cross_couple(
        fp2,
        switched_bond_tuples1,
        switched_bond_tuples2,
        forbidden_bonds=forbidden_bonds,
    )
    new_mol2, new_membership_vector2 = fp2.cross_couple(
        fp1,
        switched_bond_tuples2,
        switched_bond_tuples1,
        forbidden_bonds=forbidden_bonds,
    )

    return [new_mol1, new_mol2], [new_membership_vector1, new_membership_vector2]


def possible_fragment_sizes(
    cg, fragment_ratio_range=[0.0, 0.5], fragment_size_range=None
):
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


#   TODO add splitting by heavy-H bond?
def randomized_split_membership_vector(cg, origin_choices, fragment_size):
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
            for neigh in cg.graph.neighbors(added_id):
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
    """ """

    ppfs_kwargs = {
        "nhatoms_range": nhatoms_range,
        "fragment_ratio_range": cross_coupling_fragment_ratio_range,
        "fragment_size_range": cross_coupling_fragment_size_range,
    }

    pair_fragment_sizes = possible_pair_fragment_sizes(cg_pair, **ppfs_kwargs)

    if len(pair_fragment_sizes) == 0:
        return None, None

    tot_choice_prob_ratio = float(len(pair_fragment_sizes))

    final_pair_fragment_sizes = random.choice(pair_fragment_sizes)

    membership_vectors = []
    for cg, fragment_size in zip(cg_pair, final_pair_fragment_sizes):
        origin_choices = cg.unrepeated_atom_list()
        tot_choice_prob_ratio *= len(origin_choices)
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

    tot_choice_prob_ratio *= len(mdtuples)

    final_md_tuple = random.choice(mdtuples)

    new_cg_pair, new_membership_vectors = random_connect_fragments(
        *fragment_pairs, final_md_tuple, forbidden_bonds=forbidden_bonds
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
    tot_choice_prob_ratio /= len(
        possible_pair_fragment_sizes(new_cg_pair, **ppfs_kwargs)
    )
    # Account for probability of choosing the necessary resonance structure.
    backwards_fragment_pairs = [
        FragmentPair(new_cg, new_membership_vector)
        for new_cg, new_membership_vector in zip(new_cg_pair, new_membership_vectors)
    ]
    backwards_mdtuples = matching_dict_tuples(*backwards_fragment_pairs)

    tot_choice_prob_ratio /= len(backwards_mdtuples)

    for new_cg in new_cg_pair:
        tot_choice_prob_ratio /= len(new_cg.unrepeated_atom_list())

    return new_cg_pair, -np.log(tot_choice_prob_ratio)


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
