#TODO Perhaps add support for atoms with higher valences being added directly?
#TODO Ensure resonance structure invariance for genetic algorithms.

from xyz2mol import int_atom
import numpy as np
from .ext_graph_compound import ExtGraphCompound
from .valence_treatment import ChemGraph, default_valence, int_atom_checked, sorted_tuple, connection_forbidden,\
                            split_chemgraph_no_dissociation_check, max_bo, combine_chemgraphs
from copy import deepcopy
from sortedcontainers import SortedList
import random, itertools

def atom_equivalent_to_list_member(egc, atom_id, atom_id_list):
    for other_atom_id in atom_id_list:
        if egc.chemgraph.atom_pair_equivalent(atom_id, other_atom_id):
            return True
    return False

def atom_pair_equivalent_to_list_member(egc, atom_pair, atom_pair_list):
    for other_atom_pair in atom_pair_list:
        if egc.chemgraph.pairs_equivalent(atom_pair[:2], other_atom_pair[:2]):
            return True
    return False


# Some modification functions for EGC objects.
# TO-DO: does it make sense to get rid of them, leaving only ChemGraph functions?
def atom_replacement_possibilities(egc, inserted_atom, inserted_valence=None, replaced_atom=None, forbidden_bonds=None, exclude_equivalent=True, **other_kwargs):
    possible_ids=[]
    inserted_iac=int_atom_checked(inserted_atom)
    if replaced_atom is not None:
        replaced_iac=int_atom_checked(replaced_atom)
    if inserted_valence is None:
        inserted_valence=default_valence(inserted_iac)
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if replaced_atom is not None:
            if ha.ncharge != replaced_iac:
                continue
        if inserted_iac == ha.ncharge:
            continue
        if forbidden_bonds is not None:
            cont=False
            for neigh in egc.chemgraph.graph.neighbors(ha_id):
                if connection_forbidden(egc.nuclear_charges[neigh], inserted_atom, forbidden_bonds):
                    cont=True
                    break
            if cont:
                continue
        if ha.smallest_valid_valence()!=ha.valence:
            continue
        if ha.valence-inserted_valence<=ha.nhydrogens:
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            possible_ids.append(ha_id)
    return possible_ids

def atom_removal_possibilities(egc, deleted_atom=None, exclude_equivalent=True, only_end_atoms=False, nhatoms_range=None, **other_kwargs):
    if nhatoms_range is not None:
        if egc.num_heavy_atoms()<=nhatoms_range[0]:
            return []
    possible_ids=[]
    if deleted_atom is not None:
        deleted_iac=int_atom_checked(deleted_atom)
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if deleted_atom is not None:
            if ha.ncharge != deleted_iac:
                continue
        if ha.smallest_valid_valence()!=ha.valence:
            continue
        if only_end_atoms:
            neighs=egc.chemgraph.graph.neighbors(ha_id)
            if len(neighs)>1:
                continue
            if egc.chemgraph.aa_all_bond_orders(ha_id, neighs[0]) != [1]:
                continue
        if exclude_equivalent:
            if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                continue
        possible_ids.append(ha_id)
    return possible_ids

def chain_addition_possibilities(egc, chain_starting_element=None, forbidden_bonds=None, exclude_equivalent=True, nhatoms_range=None, **other_kwargs):
    if nhatoms_range is not None:
        if egc.num_heavy_atoms()>=nhatoms_range[1]:
            return []
    possible_ids=[]
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        if ((ha.nhydrogens>0) and (not connection_forbidden(ha.ncharge, chain_starting_element, forbidden_bonds))):
            if exclude_equivalent:
                if atom_equivalent_to_list_member(egc, ha_id, possible_ids):
                    continue
            possible_ids.append(ha_id)
    return possible_ids

def bond_change_possibilities(egc, bond_order_change, forbidden_bonds=None, fragment_member_vector=None, max_fragment_num=None, exclude_equivalent=True, **other_kwargs):
    natoms=egc.num_heavy_atoms()
    output=[]
    if bond_order_change != 0:
        for i in range(natoms):
            for j in range(i):
                bond_tuple=(j, i)
                if fragment_member_vector is not None:
                    if fragment_member_vector[i]==fragment_member_vector[j]:
                        continue
                if connection_forbidden(egc.nuclear_charges[i], egc.nuclear_charges[j], forbidden_bonds):
                    continue
                possible_bond_orders=egc.chemgraph.aa_all_bond_orders(*bond_tuple)
                max_bond_order=max(possible_bond_orders)
                if bond_order_change > 0:
                    suitable=True
                    for q in bond_tuple:
                        suitable=(suitable and (egc.chemgraph.hatoms[q].nhydrogens>=bond_order_change))
                    if suitable:
                        min_bond_order=min(possible_bond_orders)
                        suitable=(min(possible_bond_orders)+bond_order_change<=max_bo(egc.chemgraph.hatoms[i], egc.chemgraph.hatoms[j]))
                    if suitable:
                        if max_bond_order == min_bond_order:
                            possible_resonance_structures=[0]
                        else:
                            possible_resonance_structures=[egc.chemgraph.aa_all_bond_orders(*bond_tuple, unsorted=True).index(min_bond_order)]
                else:
                    suitable=(max_bond_order>=-bond_order_change)
                    if (suitable and (max_fragment_num is not None)):
                        if max_bond_order==-bond_order_change:
                            if egc.chemgraph.num_connected()==max_fragment_num:
                                suitable=(egc.chemgraph.graph.edge_connectivity(source=i, target=j) != 1)
                    if suitable:
                        # If there is a resonance structure for which decreasing bond order breaks the bond then the result of 
                        # the operation will depend on resonance structure chosen. 
                        if ( (-bond_order_change in possible_bond_orders) and (-bond_order_change != max_bond_order) and (len(possible_bond_orders) != 1) ):
                            unsorted_possible_bond_orders=egc.chemgraph.aa_all_bond_orders(*bond_tuple, unsorted=True)
                            bond_break_index=unsorted_possible_bond_orders.index(-bond_order_change)
                            max_bond_index=unsorted_possible_bond_orders.index(max_bond_order)
                            possible_resonance_structures=[bond_break_index]
                            if max_bond_index != bond_break_index:
                                possible_resonance_structures.append(max_bond_index)
                        else:
                            possible_resonance_structures=[0]
                if suitable:
                    if exclude_equivalent:
                        if atom_pair_equivalent_to_list_member(egc, bond_tuple, output):
                            continue
                    for poss_res_struct in possible_resonance_structures:
                        output.append((*bond_tuple, poss_res_struct))

    return output
    
def valence_change_possibilities(egc, exclude_equivalent=True):
    output={}
    for ha_id, ha in enumerate(egc.chemgraph.hatoms):
        cur_valence=ha.valence
        valence_list=ha.avail_val_list()
        if isinstance(valence_list, tuple):
            cur_val_id=valence_list.index(cur_valence)
            available_valences=list(valence_list[cur_val_id+1:])
            for potential_valence in valence_list[:cur_val_id]:
                if cur_valence-potential_valence<=ha.nhydrogens:
                    available_valences.append(potential_valence)
            if len(available_valences)!=0:
                if exclude_equivalent:
                    if atom_equivalent_to_list_member(egc, ha_id, output):
                        continue
                output[ha_id]=available_valences
    return output

def add_fragment_possibilities(egc, fragment, forbidden_bonds=None):
    return fragment.connection_opportunities(egc.chemgraph, forbidden_bonds=forbidden_bonds)

def add_heavy_atom_chain(egc, modified_atom_id, new_chain_atoms):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.add_heavy_atom_chain(modified_atom_id, new_chain_atoms)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def replace_heavy_atom(egc, replaced_atom_id, inserted_atom, inserted_valence=None):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.replace_heavy_atom(replaced_atom_id, inserted_atom, inserted_valence=inserted_valence)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def remove_heavy_atom(egc, removed_atom_id):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.remove_heavy_atom(removed_atom_id)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def change_bond_order(egc, atom_id1, atom_id2, bond_order_change, resonance_structure_id=0):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.change_bond_order(atom_id1, atom_id2, bond_order_change, resonance_structure_id=resonance_structure_id)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def change_valence(egc, modified_atom_id, new_valence):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.change_valence(modified_atom_id, new_valence)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def add_fragment(egc, fragment, connecting_positions):
    new_chemgraph=fragment.add_to(egc.chemgraph, connecting_positions)
    return ExtGraphCompound(chemgraph=new_chemgraph)




# Manipulating fragments.
def true_nhatoms(cg):
    return sum([ha.ncharge != 0 for ha in cg.hatoms])

def connect_fragments(frag1, frag2, connection_tuples):

    new_mol=combine_chemgraphs(frag1, frag2)

    to_delete=[]

    for connection_tuple in connection_tuples:
        vatom1=connection_tuple[0]
        vatom2=connection_tuple[1]+frag1.nhatoms()
        conn_val=new_mol.hatoms[vatom1].valence
        assert(conn_val == new_mol.hatoms[vatom2].valence)
        connected_atoms=[]
        for vatom in [vatom1, vatom2]:
            connected_atom=new_mol.neighbors(vatom)[0]
            old_tuple=sorted_tuple(vatom, connected_atom)
            new_mol.change_bond_order(*old_tuple, -conn_val)
            assert(old_tuple not in new_mol.bond_orders)
            connected_atoms.append(connected_atom)
            to_delete.append(vatom)
        new_mol.change_bond_order(*connected_atoms, conn_val)

    to_delete.sort(reverse=True)

    for i in to_delete:
        new_mol.remove_heavy_atom(i)
    return new_mol

def fragment_bond_order_dict(frag):
    output={}
    for ha_id, ha in enumerate(frag.hatoms):
        if ha.ncharge==0:
            if ha.valence in output:
                output[ha.valence].append(ha_id)
            else:
                output[ha.valence]=[ha_id]
    return output

def connected_charge(cg, vatom_id):
    return cg.hatoms[cg.neighbors(vatom_id)[0]].ncharge

def possible_connection_tuples(frag1, frag2, forbidden_bonds=None):
    frag1_bos=fragment_bond_order_dict(frag1)
    frag2_bos=fragment_bond_order_dict(frag2)

    # Size consistency checks.
    if sorted(frag1_bos.keys()) != sorted(frag2_bos.keys()):
        return []

    for valence in frag1_bos:
        if len(frag1_bos[valence]) != len(frag2_bos[valence]):
            return []

    permutation_operators=[]
    for valence in frag1_bos:
        permutation_operators.append(itertools.permutations(frag2_bos[valence]))

    connection_tuples_possibilities=[]
    for perm_lists2 in itertools.product(*permutation_operators):
        connection_tuples=[]
        abort=False
        for valence_id, (valence, connection_positions_1) in enumerate(frag1_bos.items()):
            connection_positions_2=perm_lists2[valence_id]
            for cp1, cp2 in zip(connection_positions_1, connection_positions_2):
                if forbidden_bonds is not None:
                    if sorted_tuple(connected_charge(frag1, cp1), connected_charge(frag2, cp2)) in forbidden_bonds:
                        abort=True
                        break
                connection_tuples.append((cp1, cp2))
            if abort:
                break
        if not abort:
            connection_tuples_possibilities.append(connection_tuples)

    return connection_tuples_possibilities

def all_fragment_connections(frag1, frag2, forbidden_bonds=None):
    output=[]
    connection_tuples_lists=possible_connection_tuples(frag1, frag2, forbidden_bonds=forbidden_bonds)
    for connection_tuples in connection_tuples_lists:
        new_mol=connect_fragments(frag1, frag2, connection_tuples)
        if new_mol not in output:
            output.append(new_mol)
    return output

# Procedures for genetic algorithms.

def possible_fragment_sizes(cg, fragment_ratio_range=[.0, .5], fragment_size_range=None):
    if fragment_ratio_range is None:
        bounds=deepcopy(fragment_size_range)
    else:
        bounds=[int(r*cg.nhatoms()) for r in fragment_ratio_range ]
    if bounds[1] >=cg.nhatoms():
        bounds[1]=cg.nhatoms-1
    for j in range(2):
        if bounds[j] == 0:
            bounds[j]=1
    return range(bounds[0], bounds[1]+1)

def possible_pair_fragment_sizes(cg_pair, nhatoms_range=None, **possible_fragment_sizes_kwargs):
    possible_fragment_sizes_iterators=[]
    for cg in cg_pair:
        possible_fragment_sizes_iterators.append(possible_fragment_sizes(cg, **possible_fragment_sizes_kwargs))
    output=[]
    for size_tuple in itertools.product(*possible_fragment_sizes_iterators):
        if nhatoms_range is not None:
            unfit=False
            for cg, ordered_size_tuple in zip(cg_pair, itertools.permutations(size_tuple)):
                new_mol_size=cg.nhatoms()+ordered_size_tuple[1]-ordered_size_tuple[0]
                if (new_mol_size<nhatoms_range[0]) or (new_mol_size>nhatoms_range[1]):
                    unfit=True
                    break
            if unfit:
                continue
        output.append(size_tuple)
    return output

#   TODO add splitting by heavy-H bond?
def randomized_split_membership_vector(cg, fragment_size):
    membership_vector=np.zeros(cg.nhatoms(), dtype=int)

    origin_choices=cg.unrepeated_atom_list()

    start_id=random.choice(origin_choices)

    prev_to_add=[start_id]
    remaining_atoms=fragment_size
    frag_id=1
    while ((len(prev_to_add)!=0) and (remaining_atoms!=0)):
        for added_id in prev_to_add:
            membership_vector[added_id]=frag_id
        remaining_atoms-=len(prev_to_add)
        if remaining_atoms==0:
            break
        new_to_add=[]
        for added_id in prev_to_add:
            for neigh in cg.graph.neighbors(added_id):
                if ((membership_vector[neigh]==0) and (neigh not in new_to_add)):
                    new_to_add.append(neigh)
        if len(new_to_add)>remaining_atoms:
            random.shuffle(new_to_add)
            del(new_to_add[:len(new_to_add)-remaining_atoms])
        prev_to_add=new_to_add

    return membership_vector

def split_chemgraph_by_membership_vector(cg, membership_vector):
    # Check whether some resonance structures are affected by the split.
    affected_resonance_structure_regions=[]
    resonance_structure_orders_iterators=[]
    cg.init_resonance_structures()
    for rsr_id, rsr_nodelist in enumerate(cg.resonance_structure_inverse_map):
        for i in rsr_nodelist[:-1]:
            if membership_vector[i] != membership_vector[-1]:
                affected_resonance_structure_regions.append(rsr_id)
                resonance_structure_orders_iterators.append(range(len(cg.resonance_structure_orders[rsr_id])))
                break
    # Break the molecule down into the fragments.
    frags=[]
    if len(affected_resonance_structure_regions)==0:
        frags=[split_chemgraph_no_dissociation_check(cg, membership_vector)]
    else:
        #TODO pre-check whether relevant bonds are affected?
        used_cg=deepcopy(cg)
        for resonance_structure_orders_ids in itertools.product(*resonance_structure_orders_iterators):
            for resonance_structure_region_id, resonance_structure_orders_id in zip(affected_resonance_structure_regions, resonance_structure_orders_ids):
                used_cg.adjust_resonance_valences(resonance_structure_region_id, resonance_structure_orders_id)
            new_frags=split_chemgraph_no_dissociation_check(used_cg, membership_vector)
            if new_frags not in frags:
                frags.append(new_frags)
    return frags

def all_potential_couplings(cg1, membership_vector1, cg2, membership_vector2, forbidden_bonds=None):
    frags1_list=split_chemgraph_by_membership_vector(cg1, membership_vector1)
    frags2_list=split_chemgraph_by_membership_vector(cg2, membership_vector2)
    mol_pairs=[]

    for frags1 in frags1_list:
        for frags2 in frags2_list:

            new_frag_pairs=[[frags1[0], frags2[1]], [frags2[0], frags1[1]]]
            new_mols_unpaired=[]
            membership_vectors=[]

            for new_frag_pair in new_frag_pairs:
                new_mols_unpaired_row=[]
                candidate_mols=all_fragment_connections(*new_frag_pair, forbidden_bonds=forbidden_bonds)
                backward_membership_vector=np.append(np.zeros((true_nhatoms(new_frag_pair[0]),), dtype=int), np.ones((true_nhatoms(new_frag_pair[1]),), dtype=int))
                for candidate_mol in candidate_mols:
                    added_list=[candidate_mol, backward_membership_vector]
                    if added_list not in new_mols_unpaired_row:
                        new_mols_unpaired_row.append(added_list)
                membership_vectors.append(backward_membership_vector)
                new_mols_unpaired.append(new_mols_unpaired_row)

            for added_list1, added_list2 in itertools.product(*new_mols_unpaired):
                mol_pair=[*added_list1, *added_list2]
                if mol_pair not in mol_pairs:
                    mol_pairs.append(mol_pair)

    return mol_pairs


def randomized_cross_coupling(cg_pair, cross_coupling_fragment_ratio_range=[.0, .5], cross_coupling_fragment_size_range=None, forbidden_bonds=None, nhatoms_range=None, **dummy_kwargs):

    ppfs_kwargs={"nhatoms_range" : nhatoms_range, "fragment_ratio_range" : cross_coupling_fragment_ratio_range, "fragment_size_range" : cross_coupling_fragment_size_range}
    apc_kwargs={"forbidden_bonds" : forbidden_bonds}

    pair_fragment_sizes=possible_pair_fragment_sizes(cg_pair, **ppfs_kwargs)

    if len(pair_fragment_sizes)==0:
        return None, None

    tot_choice_prob_ratio=float(len(pair_fragment_sizes))

    final_pair_fragment_sizes=random.choice(pair_fragment_sizes)

    membership_vectors=[randomized_split_membership_vector(cg, fragment_size) for cg, fragment_size in zip(cg_pair, final_pair_fragment_sizes)]

    all_pot_coup_args=[]
    for cg, membership_vector in zip(cg_pair, membership_vectors):
        all_pot_coup_args+=[cg, membership_vector]

    trial_mols=all_potential_couplings(*all_pot_coup_args, **apc_kwargs)

    if len(trial_mols)==0:
        return None, None

    tot_choice_prob_ratio*=len(trial_mols)
    final_result=random.choice(trial_mols)

    # Actual generated pair of molecules.
    new_cg_pair=[final_result[0], final_result[2]]

    # Account for probability of choosing the correct fragment sizes to do the inverse move.
    tot_choice_prob_ratio/=len(possible_pair_fragment_sizes(new_cg_pair, **ppfs_kwargs))
    # Account for probability of choosing the necessary fragment pair to do the inverse move.
    tot_choice_prob_ratio/=len(all_potential_couplings(*final_result, **apc_kwargs))

    return new_cg_pair, -np.log(tot_choice_prob_ratio)

