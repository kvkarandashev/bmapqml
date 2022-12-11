# Everything related to cross-couplings.
from .valence_treatment import (
    ChemGraph,
    sorted_by_membership,
    sorted_tuple,
    connection_forbidden,
)
import numpy as np
import random, bisect, itertools
from igraph.operators import disjoint_union
from copy import deepcopy


class FragmentPair:
    def __init__(
        self, cg: ChemGraph, membership_vector: list or np.array, affected_bonds=None
    ):
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
        self.affected_bonds = affected_bonds
        self.init_affected_bonds()
        self.init_affected_status_info()

    def init_affected_bonds(self):
        if self.affected_bonds is None:
            self.affected_bonds = []
            for i in self.sorted_vertices[0]:
                for neigh in self.chemgraph.neighbors(i):
                    if self.membership_vector[neigh] != 0:
                        self.affected_bonds.append((i, neigh))

    def init_affected_status_info(self):
        # Find resonance structures affected by the bond split.
        self.affected_resonance_structures = []
        resonance_structure_orders_iterators = []

        resonance_structure_affected_bonds = {}
        saved_all_bond_orders = {}

        default_bond_order_dict = {}

        for bond_tuple in self.affected_bonds:
            bond_stuple = sorted_tuple(*bond_tuple)
            if bond_stuple in self.chemgraph.resonance_structure_map:
                rsr_id = self.chemgraph.resonance_structure_map[bond_stuple]
                if rsr_id not in saved_all_bond_orders:
                    saved_all_bond_orders[rsr_id] = {}
                    self.affected_resonance_structures.append(rsr_id)
                    resonance_structure_orders_iterators.append(
                        range(len(self.chemgraph.resonance_structure_orders[rsr_id]))
                    )
                if rsr_id not in resonance_structure_affected_bonds:
                    resonance_structure_affected_bonds[rsr_id] = []
                resonance_structure_affected_bonds[rsr_id].append(bond_tuple)
                saved_all_bond_orders[rsr_id][
                    bond_tuple
                ] = self.chemgraph.aa_all_bond_orders(*bond_stuple, unsorted=True)

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
            self.chemgraph.hatoms[ha_id].mincopy() for ha_id in self.sorted_vertices[0]
        ] + [
            other_fp.chemgraph.hatoms[ha_id].mincopy()
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
                    new_hatoms[new_bond_tuple[0]].ncharge,
                    new_hatoms[new_bond_tuple[1]].ncharge,
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


def possible_fragment_size_bounds(
    cg, fragment_ratio_range=[0.0, 1.0], fragment_size_range=None
):
    """
    Possible fragment sizes for a ChemGraph objects satisfying enforced constraints.
    """
    if fragment_ratio_range is None:
        bounds = deepcopy(fragment_size_range)
    else:
        bounds = [int(r * cg.nhatoms()) for r in fragment_ratio_range]
    if bounds[1] >= cg.nhatoms():
        bounds[1] = cg.nhatoms() - 1
    for j in range(2):
        if bounds[j] == 0:
            bounds[j] = 1
    return bounds


# TODO: is there an analytic way to randomly choose a size tuple that does not involve all of this?
class PossiblePairFragSizesGenerator:
    def __init__(
        self, cg1, cg2, nhatoms_range=None, **possible_fragment_size_bounds_kwargs
    ):
        self.frag_size_bounds1 = possible_fragment_size_bounds(
            cg1, **possible_fragment_size_bounds_kwargs
        )
        self.frag_size_bounds2 = possible_fragment_size_bounds(
            cg2, **possible_fragment_size_bounds_kwargs
        )
        if nhatoms_range is None:
            min_diff = None
            max_diff = None
        else:
            min_diff = max(
                nhatoms_range[0] - cg1.nhatoms(), cg2.nhatoms() - nhatoms_range[1]
            )
            max_diff = min(
                nhatoms_range[1] - cg1.nhatoms(), cg2.nhatoms() - nhatoms_range[0]
            )

        num_frag_size1_poss = self.frag_size_bounds1[1] - self.frag_size_bounds1[0] + 1

        self.frag_size2_poss_numbers = np.zeros(num_frag_size1_poss, dtype=int)
        self.frag_size2_mins = np.zeros(num_frag_size1_poss, dtype=int)

        for i in range(num_frag_size1_poss):
            frag_size1 = self.frag_size_bounds1[0] + i
            frag_size2_min = max(self.frag_size_bounds2[0], frag_size1 + min_diff)
            frag_size2_max = min(self.frag_size_bounds2[1], frag_size1 + max_diff)

            self.frag_size2_poss_numbers[i] = (
                frag_size2_max
                - frag_size2_min
                + 1
                + self.frag_size2_poss_numbers[i - 1]
            )
            self.frag_size2_mins[i] = frag_size2_min

    def tot_poss_count(self):
        return self.frag_size2_poss_numbers[-1]

    def poss_frag_sizes(self, poss_id):
        frag_size1_id = bisect.bisect_right(self.frag_size2_poss_numbers, poss_id)
        if frag_size1_id != 0:
            poss_id -= self.frag_size2_poss_numbers[frag_size1_id - 1]
        return (
            self.frag_size_bounds1[0] + frag_size1_id,
            self.frag_size2_mins[frag_size1_id] + poss_id,
        )


def randomized_split_membership_vector(cg, fragment_size, origin_choices=None):
    """
    Randomly create a membership vector that can be used to split ChemGraph object into a FragmentPair object.
    """
    membership_vector = np.zeros(cg.nhatoms(), dtype=int)

    if origin_choices is None:
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
    cross_coupling_fragment_ratio_range: list or None = [0.0, 1.0],
    cross_coupling_fragment_size_range: list or None = None,
    forbidden_bonds: list or None = None,
    nhatoms_range: list or None = None,
    visited_tp_list: list or None = None,
    **dummy_kwargs,
):
    if any(cg.nhatoms() == 1 for cg in cg_pair):
        return None, None

    """
    Break two ChemGraph objects into two FragmentPair objects; the fragments are then re-coupled into two new ChemGraph objects.
    """

    ppfsg_kwargs = {
        "nhatoms_range": nhatoms_range,
        "fragment_ratio_range": cross_coupling_fragment_ratio_range,
        "fragment_size_range": cross_coupling_fragment_size_range,
    }

    PPFSG = PossiblePairFragSizesGenerator(*cg_pair, **ppfsg_kwargs)

    num_pair_fragment_sizes = PPFSG.tot_poss_count()

    if num_pair_fragment_sizes == 0:
        return None, None

    # tot_choice_prob_ratio is the ratio of probability of the trial move divided by probability of the inverse move.
    tot_choice_prob_ratio = 1.0 / num_pair_fragment_sizes

    final_pair_fragment_size_tuple_id = random.randrange(num_pair_fragment_sizes)

    final_pair_fragment_sizes = PPFSG.poss_frag_sizes(final_pair_fragment_size_tuple_id)

    membership_vectors = []
    for cg, fragment_size in zip(cg_pair, final_pair_fragment_sizes):
        origin_choices = cg.unrepeated_atom_list()
        tot_choice_prob_ratio /= len(origin_choices)
        membership_vectors.append(
            randomized_split_membership_vector(
                cg, fragment_size, origin_choices=origin_choices
            )
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
    tot_choice_prob_ratio *= PossiblePairFragSizesGenerator(
        *new_cg_pair, **ppfsg_kwargs
    ).tot_poss_count()
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
