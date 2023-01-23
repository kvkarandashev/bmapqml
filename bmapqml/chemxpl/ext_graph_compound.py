# TO-DO: do not call readjust_graph and do not check valences in __init__ function. Make a separate constructor that does it automatically?
# TO-DO: Store the graph of sigma bonds (without order) in "sigma graph" with the default graph generated from default adjacency matrix?
#           Otherwise we'll have to call readjust_graph each time we change a molecule.

from g2s import GraphCompound
from g2s.utils import calculate_distances
import numpy as np
from .valence_treatment import ChemGraph
from .periodic import max_ecn
import math


class ExtGraphCompound(GraphCompound):
    def __init__(
        self,
        adjacency_matrix=None,
        nuclear_charges=None,
        chemgraph=None,
        coordinates=None,
        distances=None,
        elements=None,
        hydrogen_autofill=False,
        bond_orders=None,
        additional_data={},
    ):
        """
        An "extended" inheritor to the GraphCompound object.
        """
        self.chemgraph = chemgraph
        if self.chemgraph is None:
            if (nuclear_charges is not None) and (
                (adjacency_matrix is not None) or (bond_orders is not None)
            ):
                self.chemgraph = ChemGraph(
                    adj_mat=adjacency_matrix,
                    nuclear_charges=nuclear_charges,
                    hydrogen_autofill=hydrogen_autofill,
                    bond_orders=bond_orders,
                )
        if (nuclear_charges is None) or hydrogen_autofill:
            nuclear_charges = self.chemgraph.full_ncharges()
        # Sometimes rdkit determines bond orders wrong; by default try to correct them.
        self.coordinates = coordinates
        if distances is None:
            if self.coordinates is not None:
                distances = calculate_distances(self.coordinates)
        if (adjacency_matrix is None) or hydrogen_autofill:
            adjacency_matrix = self.chemgraph.full_adjmat()
        super().__init__(adjacency_matrix, nuclear_charges, distances=distances)
        # Double check that all bond orders are correct (sometimes molecular graphs are not properly imported).
        for atom_id1 in range(self.chemgraph.nhatoms()):
            for atom_id2 in range(self.chemgraph.nhatoms()):
                self.adjacency_matrix[atom_id1, atom_id2] = self.chemgraph.bond_order(
                    atom_id1, atom_id2
                )
        # In case we want to attach more data to the same entry.
        self.additional_data = additional_data

    #   Largely copies what's done in G2S, but make sure coordinates are changed too.
    #   It also helps a lot to sort by just moving hydrogens to the end of the self.nuclear_charges without shuffling the heavy atoms.
    #   TO-DO synchronize with G2S somehow?
    def resort_atoms(self):
        self.resort_atom_idx = np.empty(len(self.nuclear_charges), dtype=int)
        heavy_atom_id = 0
        hydrogen_id = len(self.nuclear_charges) - 1
        for atom_id, atom_charge in enumerate(self.nuclear_charges):
            if atom_charge == 1:
                self.resort_atom_idx[hydrogen_id] = atom_id
                hydrogen_id -= 1
            else:
                self.resort_atom_idx[heavy_atom_id] = atom_id
                heavy_atom_id += 1
        self.adjacency_matrix = self.adjacency_matrix[self.resort_atom_idx][
            :, self.resort_atom_idx
        ]
        self.nuclear_charges = self.nuclear_charges[self.resort_atom_idx]
        if self.distances is not None:
            self.distances = self.distances[self.resort_atom_idx][
                :, self.resort_atom_idx
            ]
        if self.coordinates is not None:
            self.coordinates = self.coordinates[self.resort_atom_idx]

    #   A copy of what is done in G2S with an extra option to allow sorting in the other order (makes more sense for Coulomb matrices, I think).
    def sort_norm(
        self, descending=False, canonical_ordering_sort=False, norm_added_delta=None
    ):
        # TO-DO do not do sorting of coordinates here, initialize inv_sorting_idx and use it in geometry_krr
        """
        Resort representation, nuclear charges, adjacency matrix and distance matrix using the
        row norm of the representation.
        Ensure permutational invariance!
        """
        if canonical_ordering_sort:
            if self.chemgraph.inv_canonical_permutation is None:
                self.chemgraph.init_canonical_permutation()
            idx_list = self.chemgraph.inv_canonical_permutation
        else:
            row_norms = np.linalg.norm(self.representation, axis=1)
            if norm_added_delta is not None:
                if self.chemgraph.canonical_permutation is None:
                    self.chemgraph.init_canonical_permutation()
                row_norms -= norm_added_delta * np.array(
                    self.chemgraph.canonical_permutation
                )
            if descending:
                row_norms *= -1
            idx_list = np.argsort(row_norms)
        self.sorting_idxs = idx_list
        self.representation = self.representation[idx_list][:, idx_list]
        self.nuclear_charges = self.nuclear_charges[idx_list]
        self.adjacency_matrix = self.adjacency_matrix[idx_list][:, idx_list]
        if self.filtered:
            atom_idxs = np.arange(
                len(self.nuclear_charges), len(self.full_nuclear_charges)
            )
            full_idx = np.array([*idx_list, *atom_idxs])
            self.full_adjacency_matrix = self.full_adjacency_matrix[full_idx][
                :, full_idx
            ]
            self.full_nuclear_charges = self.full_nuclear_charges[full_idx]
        if self.distances is not None:
            self.distances = self.distances[idx_list][:, idx_list]
            if self.filtered:
                self.full_distances = self.full_distances[full_idx][:, full_idx]

    # This is a modification of Dominik's "graph Coulomb matrix" representation that should be more suitable for non-structural properties
    # predictions and MIGHT solve some problems I've been seeing with geometry prediction.
    def generate_res_av_coulomb_matrix(
        self,
        size=9,
        descending_sorting=True,
        sort_up_to_eff_coord_num=None,
        canonical_ordering_sort=False,
        use_bond_order_dists=False,
        norm_added_delta=1e-6,
    ):
        dist_mats = self.get_res_av_dist_mats(
            sort_up_to_eff_coord_num=sort_up_to_eff_coord_num,
            use_bond_order_dists=use_bond_order_dists,
        )
        # Construct Coulomb matrices. TO-DO: make fancy references to numpy procedures?
        for dm_id in range(len(dist_mats)):
            for i in range(self.num_heavy_atoms()):
                dist_mats[dm_id][i, i] = 0.5 * self.nuclear_charges[i] ** 2.4
                for j in range(i):
                    if math.isinf(dist_mats[dm_id][i, j]):
                        dist_mats[dm_id][i, j] = 0.0
                    else:
                        dist_mats[dm_id][i, j] **= -1
                        dist_mats[dm_id][i, j] *= (
                            self.nuclear_charges[i] * self.nuclear_charges[j]
                        )
                    dist_mats[dm_id][j, i] = dist_mats[dm_id][i, j]
        self.representation = dist_mats[0]
        self.sort_norm(
            descending=descending_sorting,
            canonical_ordering_sort=canonical_ordering_sort,
            norm_added_delta=norm_added_delta,
        )
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(size, k=0)]
        self.augment_representation(dist_mats[1:], size)

    def generate_res_av_md_coulomb_matrix(
        self,
        size=9,
        descending_sorting=True,
        sort_up_to_eff_coord_num=None,
        canonical_ordering_sort=True,
        use_bond_order_dists=False,
        norm_added_delta=1e-6,
    ):
        #
        atom_multipliers = np.array(
            [list(ha.comparison_iterator()) for ha in self.chemgraph.hatoms]
        ).T
        dist_mats = self.get_res_av_dist_mats(
            sort_up_to_eff_coord_num=sort_up_to_eff_coord_num,
            use_bond_order_dists=use_bond_order_dists,
        )
        self.representation = np.zeros(
            (self.num_heavy_atoms(), self.num_heavy_atoms()), dtype=float
        )
        rep_components = []
        for dist_mat in dist_mats:
            inv_dist_mat = np.eye(self.num_heavy_atoms(), dtype=float)
            for i in range(self.num_heavy_atoms()):
                for j in range(i):
                    inv_dist_mat[i, j] = 1.0 / dist_mat[i, j]
                    inv_dist_mat[j, i] = inv_dist_mat[i, j]
            for am in atom_multipliers:
                cur_rep_comp = inv_dist_mat * am
                if canonical_ordering_sort is False:
                    self.representation += cur_rep_comp**2
                rep_components.append(cur_rep_comp)
        self.sort_norm(
            descending=descending_sorting,
            canonical_ordering_sort=canonical_ordering_sort,
            norm_added_delta=norm_added_delta,
        )
        self.zero_padding(size)
        self.representation = np.array([], dtype=float)
        self.augment_representation(rep_components, size, matrices_asymmetric=True)

    def augment_representation(self, extra_mats, size, k=1, matrices_asymmetric=False):
        if len(extra_mats) != 0:
            temp_mat = np.zeros((size, size), dtype=float)
            for extra_mat in extra_mats[1:]:
                temp_mat[
                    : self.num_heavy_atoms(), : self.num_heavy_atoms()
                ] = extra_mat[self.sorting_idxs][:, self.sorting_idxs]
                if matrices_asymmetric:
                    self.representation = np.append(
                        self.representation, np.reshape(temp_mat, (size**2,))
                    )
                else:
                    self.representation = np.append(
                        self.representation, temp_mat[np.triu_indices(size, k=k)]
                    )

    def generate_res_av_bond_length(
        self,
        size=9,
        sort_up_to_eff_coord_num=None,
        canonical_ordering_sort=False,
        norm_added_delta=1e-6,
    ):
        dist_mats = self.get_res_av_dist_mats(
            sort_up_to_eff_coord_num=sort_up_to_eff_coord_num
        )
        self.representation = dist_mats[0]
        self.sort_norm(
            canonical_ordering_sort=canonical_ordering_sort,
            norm_added_delta=norm_added_delta,
        )
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(size, k=1)]
        self.augment_representation(dist_mats[1:], size)

    def get_res_av_dist_mats(
        self, sort_up_to_eff_coord_num=None, use_bond_order_dists=False
    ):
        edge_list = self.chemgraph.graph.get_edgelist()
        if use_bond_order_dists:
            bond_lengths = 1.0 / self.chemgraph.get_res_av_bond_orders(
                edge_list=edge_list
            )
        else:
            bond_lengths = self.chemgraph.get_res_av_bond_lengths(edge_list=edge_list)
        output = [self.chemgraph.shortest_paths(weights=bond_lengths)]
        if sort_up_to_eff_coord_num is not None:
            ecn_list = [
                self.chemgraph.effective_coordination_number(ha_id)
                for ha_id in range(self.num_heavy_atoms())
            ]
            for cur_max_ecn in range(max_ecn - 1, sort_up_to_eff_coord_num - 1, -1):
                for eid, bl in enumerate(bond_lengths):
                    if not math.isinf(bl):
                        for ha_id in edge_list[eid]:
                            if ecn_list[ha_id] > cur_max_ecn:
                                bond_lengths[eid] = float("inf")
                                break
                output.append(self.chemgraph.shortest_paths(weights=bond_lengths))
        return output

    # TO-DO part of an eariler draft, might be useful later.
    def rep_combine_sort_zero_padd(self, representation_mats, size=9):
        inv_canonical_permutation = self.chemgraph.get_inv_canonical_permutation()
        # Re-index all distances accordingly.
        if self.distances is not None:
            padded_distances = np.zeros((size, size), dtype=float)
            padded_distances[
                : self.num_heavy_atoms(), : self.num_heavy_atoms()
            ] = self.distances[inv_canonical_permutation][:, inv_canonical_permutation]
            self.distances = padded_distances[np.triu_indices(size, k=1)]
        self.nuclear_charges = self.nuclear_charges[inv_canonical_permutation]
        sorted_representation_matrices = np.zeros(
            (len(representation_mats), size, size), dtype=float
        )
        for i, rep_mat in enumerate(representation_mats):
            sorted_representation_matrices[i][
                : self.num_heavy_atoms(), : self.num_heavy_atoms()
            ] = rep_mat[inv_canonical_permutation][:, inv_canonical_permutation]
        self.representation = np.reshape(
            sorted_representation_matrices,
            np.product(sorted_representation_matrices.shape),
        )

    # Part of earlier draft for generate_extended_graph_coulomb_matrix
    def effective_coordination_numbers(self):
        output = np.zeros(self.num_heavy_atoms, dtype=int)
        for bond_tuple, bond_order in self.bond_orders.items():
            for hatom_id in bond_tuple:
                output[hatom_id] = max(output[hatom_id], bond_order)
        return output[hatom_id]

    def add_canon_rdkit_coords(self, canon_rdkit_coords):
        self.coordinates = np.zeros(canon_rdkit_coords.shape, dtype=float)
        self.chemgraph.init_canonical_permutation()
        for hatom_id in range(self.chemgraph.nhatoms()):
            self.coordinates[hatom_id][:] = canon_rdkit_coords[
                self.chemgraph.inv_canonical_permutation[hatom_id]
            ][:]
        hydrogen_search_lower_bounds = np.zeros(self.chemgraph.nhatoms(), dtype=int)
        cur_hydrogen_id = self.chemgraph.nhatoms()
        for connected_hatom in self.chemgraph.inv_canonical_permutation:
            for _ in range(self.chemgraph.hatoms[connected_hatom].nhydrogens):
                internal_hydrogen_id = self.find_connected_hydrogen(
                    connected_hatom,
                    start_hydrogen_id=hydrogen_search_lower_bounds[connected_hatom],
                )
                hydrogen_search_lower_bounds[connected_hatom] = internal_hydrogen_id + 1
                self.coordinates[internal_hydrogen_id][:] = canon_rdkit_coords[
                    cur_hydrogen_id
                ][:]
                cur_hydrogen_id += 1

    # TO-DO: check whether this function reappears elsewhere.
    def find_connected_hydrogen(self, hatom_id, start_hydrogen_id=0):
        row_adjmat = self.true_adjmat()[hatom_id]
        for i, (bo, nc) in enumerate(
            zip(
                row_adjmat[start_hydrogen_id:], self.nuclear_charges[start_hydrogen_id:]
            )
        ):
            if (nc == 1) and (bo == 1):
                return i + start_hydrogen_id
        return None

    def is_connected(self):
        return self.chemgraph.is_connected()

    def num_connected(self):
        return self.chemgraph.num_connected()

    def fragment_member_vector(self):
        return self.chemgraph.graph.components().membership

    def num_heavy_atoms(self):
        return self.chemgraph.nhatoms()

    def num_atoms(self):
        return self.chemgraph.full_natoms()

    def true_ncharges(self):
        if self.filtered:
            ncharges = self.full_nuclear_charges
        else:
            ncharges = self.nuclear_charges
        return np.copy(ncharges)

    def true_adjmat(self):
        if self.filtered:
            output = self.full_adjacency_matrix
        else:
            output = self.adjacency_matrix
        return np.copy(output)

    def true_distances(self):
        if self.filtered:
            output = self.full_adjacency_matrix
        else:
            output = self.adjacency_matrix
        return np.copy(output)

    def __lt__(self, egc2):
        return self.chemgraph < egc2.chemgraph

    def __gt__(self, egc2):
        return self.chemgraph > egc2.chemgraph

    def __eq__(self, egc2):
        if not isinstance(egc2, ExtGraphCompound):
            return False
        return self.chemgraph == egc2.chemgraph

    def __str__(self):
        return str(self.chemgraph)

    def __repr__(self):
        return str(self)
