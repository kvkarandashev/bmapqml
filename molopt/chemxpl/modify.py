#TO-DO 1. make everything here invariant w.r.t. resonanse structures.
# 2. Perhaps add support for atoms with higher valences being added directly?


from xyz2mol import int_atom
import numpy as np
from .ext_graph_compound import ExtGraphCompound
from .valence_treatment import ChemGraph, default_valence, int_atom_checked, Fragment, sorted_tuple, connection_forbidden, split_chemgraph_no_dissociation_check
from copy import deepcopy
from sortedcontainers import SortedList
import random

def atom_equivalent_to_list_member(egc, atom_id, atom_id_list):
    for other_atom_id in atom_id_list:
        if egc.chemgraph.atom_pair_equivalent(atom_id, other_atom_id):
            return True
    return False

def atom_pair_equivalent_to_list_member(egc, atom_pair, atom_pair_list):
    for other_atom_pair in atom_pair_list:
        if egc.chemgraph.pairs_equivalent(atom_pair, other_atom_pair):
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
            if egc.chemgraph.bond_order(ha_id, neighs[0]) != 1:
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
                if fragment_member_vector is not None:
                    if fragment_member_vector[i]==fragment_member_vector[j]:
                        continue
                if connection_forbidden(egc.nuclear_charges[i], egc.nuclear_charges[j], forbidden_bonds):
                    continue
                if bond_order_change > 0:
                    suitable=True
                    for q in [i, j]:
                        suitable=(suitable and (egc.chemgraph.hatoms[q].nhydrogens>=bond_order_change))
                else:
                    suitable=(egc.chemgraph.bond_order(j, i)>=-bond_order_change)
                    if (suitable and (max_fragment_num is not None)):
                        if egc.chemgraph.bond_order(j, i)==-bond_order_change:
                            if egc.chemgraph.num_connected()==max_fragment_num:
                                suitable=(egc.chemgraph.graph.edge_connectivity(source=i, target=j) != 1)
                if suitable:
                    if exclude_equivalent:
                        if atom_pair_equivalent_to_list_member(egc, (j, i), output):
                            continue
                    output.append((j, i))
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

def change_bond_order(egc, atom_id1, atom_id2, bond_order_change):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.change_bond_order(atom_id1, atom_id2, bond_order_change)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def change_valence(egc, modified_atom_id, new_valence):
    new_chemgraph=deepcopy(egc.chemgraph)
    new_chemgraph.change_valence(modified_atom_id, new_valence)
    return ExtGraphCompound(chemgraph=new_chemgraph)

def add_fragment(egc, fragment, connecting_positions):
    new_chemgraph=fragment.add_to(egc.chemgraph, connecting_positions)
    return ExtGraphCompound(chemgraph=new_chemgraph)



# Procedures for genetic algorithms.

def randomized_split_chemgraph(cg, num_fragmentations=1, fragment_ratio_range=None, fragment_size_range=None):
    if fragment_ratio_range is None:
        # We go by fragment size range.
        fragment_size=random.randrange(fragment_size_range[0], fragment_size_range[1]+1)
    else:
        fragment_size=int(random.uniform(*fragment_ratio_range)*cg.nhatoms())+1
    if fragment_size >= cg.nhatoms():
        fragment_size=cg.nhatoms()-1
    membership_vector=np.zeros(cg.nhatoms(), dtype=int)
    start_id=random.randrange(cg.nhatoms())
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
    # Break the molecule down into the fragments.
    frags=split_chemgraph_no_dissociation_check(cg, membership_vector)
    choice_prob=float(cg.atom_multiplicity(start_id))/cg.nhatoms()
    frag_seed_atom=np.where(np.where(membership_vector==frag_id)[0]==start_id)[0][0]
    return frags, choice_prob, frag_seed_atom

def random_fragment_pair_combination(frag1, frag2, forbidden_bonds=None):
    possible_output_mols=frag2.all_connections_with_frag(frag1, forbidden_bonds=forbidden_bonds)
    if len(possible_output_mols)==0:
        return None, None
    else:
        final_output_mol=random.choice(possible_output_mols)
        return final_output_mol, len(possible_output_mols)

def randomized_cross_coupling(cg_pair, cross_coupling_fragment_ratio_range=None, cross_coupling_fragment_size_range=None, forbidden_bonds=None, **dummy_kwargs):
    tot_choice_prob_ratio=1.0
    fragment_pairs=[]
    frag_seed_atoms=[]
    for cg in cg_pair:
        fragment_pair, choice_prob, frag_seed_atom=randomized_split_chemgraph(cg, fragment_ratio_range=cross_coupling_fragment_ratio_range, fragment_size_range=cross_coupling_fragment_size_range)
        tot_choice_prob_ratio*=choice_prob*len(fragment_pair[0].all_connections_with_frag(fragment_pair[1], forbidden_bonds=forbidden_bonds))
        fragment_pairs.append(fragment_pair)
        frag_seed_atoms.append(frag_seed_atom)
    if fragment_pairs[0][0].bo_list() != fragment_pairs[1][0].bo_list():
        return None, None
    new_frag_pairs_seeds=[[(fragment_pairs[0][0], fragment_pairs[1][1]), frag_seed_atoms[1]], [(fragment_pairs[1][0], fragment_pairs[0][1]), frag_seed_atoms[0]]]
    output=[]
    for [new_frag_pair, frag_seed_atom] in new_frag_pairs_seeds:
        new_mol_choices=new_frag_pair[1].all_connections_with_frag(new_frag_pair[0], forbidden_bonds=forbidden_bonds)
        if len(new_mol_choices)==0:
            return None, None
        new_mol=random.choice(new_mol_choices)
        # For detailed balance, get the inverse choice probability.
        reverse_seed_atom_id=frag_seed_atom+new_frag_pair[0].chemgraph.nhatoms()
        tot_choice_prob_ratio*=float(new_mol.nhatoms())/new_mol.atom_multiplicity(reverse_seed_atom_id)/len(new_mol_choices)
        output.append(new_mol)

    return output, np.log(tot_choice_prob_ratio)

