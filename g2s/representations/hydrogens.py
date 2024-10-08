import igraph
import numpy as np

from .representations import generate_bond_length


def local_bondlength(
    adjacency_matrix, nuclear_charges, distances=None, n_neighs=5, local_env=True
):
    """
    Generates a local bond length representation for hydrogen atoms.
    Only 4 heavy atom distances are considered.
    The number of neighbours considered in the representation can be tuned via the n_neighs argument.

    On top of the representation, also computes mapping of each heavy atom to hydrogens.
    The mapping has the following look:
        tuple: (central_atom_id, attached_hydrogen_ids, neighbour_ids)

    Parameters
    ----------
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix of the system.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.
    distances: np.array, shape(n_atoms, n_atoms)
        Interatomic distance matrix.
    n_neighs: int (default=5)
        Number of neighbours to include in the representation. Will always apply zero-padding if not enough
        neighbours are available.
    local_env: bool (default=True)
        If True, the hydrogen representation only includes bond length distances of heavy atoms to a target hydrogen.
        If False, the representation includes bond length distances of all neighboring atoms to all.

    Returns
    -------
    local_h_repr: np.array, shape (n_hydrogens, 4)
        Local bond length representation.
    heavy_hydrogen_mapping: np.array of tuples
        tuple: (central_atom_id, attached_hydrogen_ids, neighbour_ids)
    hydrogen_heavy_distances: np.array, shape (n_hydrogens, 5)
        Distances of closest 4 heavy atoms to a hydrogen. Last distance is the distance between two hydrogens
        in case multiple hydrogens will need to be attached.
    """
    for a in adjacency_matrix:
        if np.sum(a) == 0:
            raise AssertionError("Non Bonded Hydrogen Found!")
    n_heavy_atoms = len(np.where(nuclear_charges != 1)[0])
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_h_idx = get_hydrogen_neighbours(graph, nuclear_charges)
    bond_length_matrix = generate_bond_length(adjacency_matrix, nuclear_charges)

    local_h_repr = []
    heavy_hydrogen_mapping = []
    hydrogen_heavy_distances = []
    for heavy_atom_index, hydrogen_indices in bond_h_idx:
        # Get local atomic representation of the atom bound to hydrogen
        atomic_representation = bond_length_matrix[hydrogen_indices[0], :n_heavy_atoms]

        # Sort by bond length distances to get the closest neighbours
        # Results in a representation vector of size 4
        closest_neighbour_sorting_idx = np.argsort(atomic_representation)[:4]
        neighbour_sorting_idx = np.argsort(atomic_representation)[:n_neighs]

        if local_env:
            padded_representation = np.zeros(n_neighs + 1)
            n_hneighs = neighbour_sorting_idx.shape[0]
            padded_representation[:n_hneighs] = atomic_representation[
                neighbour_sorting_idx
            ]
            local_h_repr.append(padded_representation)
        else:
            atomic_env_idxs = np.array([hydrogen_indices[0], *neighbour_sorting_idx])
            atomic_representation = bond_length_matrix[atomic_env_idxs][
                :, atomic_env_idxs
            ]
            norm_sort_idx = np.argsort(np.linalg.norm(atomic_representation, axis=1))
            atomic_representation = atomic_representation[norm_sort_idx][
                :, norm_sort_idx
            ]
            n_hneighs = atomic_representation.shape[0]
            padded_representation = np.zeros((n_neighs + 1, n_neighs + 1))
            padded_representation[:n_hneighs, :n_hneighs] = atomic_representation
            local_h_repr.append(
                padded_representation[
                    np.triu_indices(padded_representation.shape[1], k=1)
                ]
            )

        if distances is not None:
            h_distances, hydrogen_mapping = map_closest_distance(
                heavy_atom_index,
                hydrogen_indices,
                distances,
                closest_neighbour_sorting_idx,
            )
            heavy_hydrogen_mapping.append(hydrogen_mapping)
            hydrogen_heavy_distances.append(h_distances)
        else:
            heavy_hydrogen_mapping.append(
                (heavy_atom_index, hydrogen_indices, closest_neighbour_sorting_idx[1:])
            )

    if distances is not None:
        return (
            np.array(local_h_repr),
            np.array(heavy_hydrogen_mapping),
            np.array(hydrogen_heavy_distances),
        )
    else:
        return np.array(local_h_repr), np.array(heavy_hydrogen_mapping)


def get_hydrogen_neighbours(graph, nuclear_charges):
    """
    Get all heavy atoms that have hydrogen attached
    and their corresponding indices.

    Parameters
    ----------
    graph: igraph object
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.

    Returns
    -------
    bond_h_idx: list
        Contains tuples of (heavy_atom_idx, hydrogen_idxs).

    """
    heavy_atom_idx = np.where(nuclear_charges != 1)[0]

    bond_h_idx = []
    for idx in heavy_atom_idx:
        neighs = graph.neighbors(idx, mode="IN")
        h_ids = []
        for n in neighs:
            if nuclear_charges[n] == 1:
                h_ids.append(n)
        if h_ids:
            bond_h_idx.append((idx, np.array(h_ids)))
    return bond_h_idx


def map_closest_distance(
    heavy_atom_index, hydrogen_indices, distances, neighbour_sorting_idx
):
    """
    Find hydrogen with the closest distance to all heavy atoms.

    Parameters
    ----------
    heavy_atom_index: int
        Index of the central heavy atom.
    hydrogen_indices: np.array, shape(n_hydrogens)
        Indices of hydrogens that are attached to a specific atom.
    distances: np.array, shape(n_atoms, n_atoms)
        Full distance matrix.
    neighbour_sorting_idx: np.array, shape(4)
        Indices of the closest 4 heavy atoms.

    Returns
    -------
    h_distances: np.array
    hydrogen_mapping: tuple
        (central_atom_id, attached_hydrogen_ids, neighbour_ids)
    """
    shortest_h_distances = distances[hydrogen_indices][:, neighbour_sorting_idx].sum(
        axis=1
    )
    closest_h_idx = hydrogen_indices[np.argmin(shortest_h_distances)]
    h_distances = [distances[heavy_atom_index, closest_h_idx]]
    for nbh in neighbour_sorting_idx[1:]:
        h_distances.append(distances[nbh, closest_h_idx])
    if len(hydrogen_indices) > 1:
        h_distances.append(distances[hydrogen_indices[0], hydrogen_indices[1]])
    else:
        h_distances.append(0.0)

    hydrogen_mapping = (
        heavy_atom_index,
        np.array(hydrogen_indices)[np.argsort(shortest_h_distances)],
        neighbour_sorting_idx[1:],
    )
    return np.array(h_distances), hydrogen_mapping
