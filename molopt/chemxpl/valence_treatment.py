# TO-DO: Perhaps in the future everything related to graph manipulation should 
# be included into ExtGraphCompound instead.

# TO-DO Perhaps defining "canonically_permuted" inside a ChemGraph can simplify some expressions.

import itertools, copy
import numpy as np
from igraph import Graph
from igraph.operators import disjoint_union
from .periodic import valences_int, period_int, s_int, p_int, unshared_pairs, coord_num_hybrid, max_ecn
from g2s.constants import periodic_table, atom_radii
from sortedcontainers import SortedList

try:
    from xyz2mol import int_atom
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install xyz2mol software in order to use this parser. '
                      'Visit: https://github.com/jensengroup/xyz2mol')

class InvalidAdjMat(Exception):
    pass

class InvalidChange(Exception):
    pass

# Introduced in case we, for example, started to consider F as a default addition instead.
DEFAULT_ELEMENT=1

# To avoid equality expressions for two reals.
irrelevant_bond_order_difference=1.0e-8

# Check that atom_id is integer rather than string element representation.
def int_atom_checked(atom_id):
    if isinstance(atom_id, str):
        return int_atom(atom_id)
    else:
        return atom_id

# Default valence for a given element.
def default_valence(atom_id):
    val_list=valences_int[int_atom_checked(atom_id)]
    if isinstance(val_list, tuple):
        return val_list[0]
    else:
        return val_list

# Sorted a tuple either by its value or by value of ordering tuple.
def sorted_tuple(*orig_tuple, ordering_tuple=None):
    if ordering_tuple is None:
        return tuple(sorted(orig_tuple))
    else:
        temp_list=[(i, io) for i, io in zip(orig_tuple, ordering_tuple)]
        temp_list.sort(key=lambda x : x[1])
        return tuple([i for (i, io) in temp_list])

# Sort several tuples.
def sorted_tuples(*orig_tuples):
    output=[]
    for orig_tuple in orig_tuples:
        output.append(sorted_tuple(*orig_tuple))
    return sorted(output)

# Color obj_list in a way that each equal obj1 and obj2 were the same color.
# Used for defining canonical permutation of a graph.
def list2colors(obj_list):
    ids_objs=list(enumerate(obj_list))
    ids_objs.sort(key=lambda x: x[1])
    num_obj=len(ids_objs)
    colors=np.zeros(num_obj, dtype=int)
    cur_color=0
    prev_obj=ids_objs[0][1]
    for i in range(1, num_obj):
        cur_obj=ids_objs[i][1]
        if cur_obj != prev_obj:
            cur_color+=1
            prev_obj=cur_obj
        colors[ids_objs[i][0]]=cur_color
    return colors

# To auxiliary "augmented" gt functions helpful for defining sorting inside a list.
def triple_gt(obj1, obj2):
    if obj1==obj2:
        return None
    else:
        return bool((obj1>obj2)) # To make sure it's bool and not, for example, numpy.bool_.

def triple_gt_witer(obj1, obj2):
    for i1, i2 in zip(obj1.comparison_iterator(), obj2.comparison_iterator()):
        output=triple_gt(i1, i2)
        if output is not None:
            return output
    return None

# Sort a list into several lists by membership.
def sorted_by_membership(l, membership_vector):
    n=max(membership_vector)
    output=[[] for i in range(n+1)]
    for val, m in zip(l, membership_vector):
        output[m].append(val)
    return output

# Auxiliary class mainly used to keep valences in check.
class HeavyAtom:
    def __init__(self, atom_symbol, valence=None, nhydrogens=0, coordination_number=None):
        self.ncharge=int_atom_checked(atom_symbol)
        if valence is None:
            valence=self.smallest_valid_valence(coordination_number=coordination_number)
        self.valence=valence
        self.nhydrogens=nhydrogens
    # Valence-related.
    def avail_val_list(self):
        return valences_int[self.ncharge]
    def valence_reasonable(self):
        val_list=self.avail_val_list()
        if isinstance(val_list, tuple):
            return (self.valence in val_list)
        else:
            return (self.valence==val_list)
    def smallest_valid_valence(self, coordination_number=None, show_id=False):
        val_list=self.avail_val_list()
        if isinstance(val_list, tuple):
            needed_id=None
            if coordination_number is None:
                needed_id=0
            else:
                for i, valence in enumerate(val_list):
                    if valence>=coordination_number:
                        needed_id=i
                        break
                if needed_id is None:
                    raise InvalidAdjMat
            if show_id:
                return needed_id
            else:
                return val_list[needed_id]
        else:
            if coordination_number is not None:
                if coordination_number>val_list:
                    raise InvalidAdjMat
            if show_id:
                return -1
            else:
                return val_list
    def valence_val_id(self):
        vals=self.avail_val_list()
        if isinstance(vals, tuple):
            return self.avail_val_list().index(self.valence)
        else:
            return 0
    def max_valence(self):
        val_list=self.avail_val_list()
        if isinstance(val_list, tuple):
            return val_list[-1]
        else:
            return val_list
    # Procedures for ordering.
    def comparison_iterator(self):
        return iter([self.valence-self.nhydrogens, s_int[self.ncharge], p_int[self.ncharge], period_int[self.ncharge], self.valence_val_id()])
    def __lt__(self, ha2):
        return (triple_gt_witer(self, ha2) is False)
    def __gt__(self, ha2):
        return (triple_gt_witer(self, ha2) is True)
    def __eq__(self, ha2):
        return (triple_gt_witer(self, ha2) is None)
    # Procedures for printing.
    def __str__(self):
        return "C:"+str(self.ncharge)+",V:"+str(self.valence)+",H:"+str(self.nhydrogens)
    def __repr__(self):
        return str(self)

# Functions that should help defining a meaningful distance measure between Heavy_Atom objects. TO-DO: Still need those?

def hatom_state_coords(ha):
    return [period_int[ha.ncharge], s_int[ha.ncharge], p_int[ha.ncharge], ha.valence, ha.nhydrogens]

num_state_coords={hatom_state_coords : 5}

# TO-DO perhaps used SortedDict from sortedcontainers more here?
class ChemGraph:
    def __init__(self, graph=None, hatoms=None, bond_orders=None, all_bond_orders=None, adj_mat=None, nuclear_charges=None, hydrogen_autofill=False):
        self.graph=graph
        self.hatoms=hatoms
        self.bond_orders=bond_orders
        self.all_bond_orders=all_bond_orders
        if ((self.all_bond_orders is None) and (self.bond_orders is None)):
            self.adj_mat2all_bond_orders(adj_mat)
        if ((self.graph is None) or (self.hatoms is None)):
            self.init_graph_natoms(np.array(nuclear_charges), hydrogen_autofill=hydrogen_autofill)
        self.changed()
    def adj_mat2all_bond_orders(self, adj_mat):
        self.all_bond_orders={}
        for atom1, adj_mat_row in enumerate(adj_mat):
            for atom2, adj_mat_val in enumerate(adj_mat_row[:atom1]):
                if adj_mat_val != 0:
                    self.all_bond_orders[(atom2, atom1)]=adj_mat_val
    def init_graph_natoms(self, nuclear_charges, hydrogen_autofill=False):
        self.hatoms=[]
        if self.bond_orders is not None:
            heavy_bond_orders=self.bond_orders
        self.bond_orders={}
        heavy_atom_dict={}
        for atom_id, ncharge in enumerate(nuclear_charges):
            if ncharge != DEFAULT_ELEMENT:
                heavy_atom_dict[atom_id]=len(self.hatoms)
                self.hatoms.append(HeavyAtom(ncharge, valence=0))
        self.graph=Graph(n=len(self.hatoms), directed=False)
        if hydrogen_autofill:
            filled_bond_orders=heavy_bond_orders
        else:
            filled_bond_orders=self.all_bond_orders
        for bond_tuple, bond_order in filled_bond_orders.items():
            for ha_id1, ha_id2 in itertools.permutations(bond_tuple):
                if ha_id1 in heavy_atom_dict:
                    true_id=heavy_atom_dict[ha_id1]
                    self.hatoms[true_id].valence+=bond_order
                    if ha_id2 in heavy_atom_dict:
                        if (ha_id1<ha_id2):
                            self.change_edge_order(true_id, heavy_atom_dict[ha_id2], bond_order)
                    else:
                        self.hatoms[true_id].nhydrogens+=bond_order
        if hydrogen_autofill:
            for ha_id, hatom in enumerate(self.hatoms):
                cur_assigned_valence=hatom.valence
                self.hatoms[ha_id].valence=hatom.smallest_valid_valence(coordination_number=cur_assigned_valence)
                self.hatoms[ha_id].nhydrogens+=self.hatoms[ha_id].valence-cur_assigned_valence
        # Check that valences make sense.
        if not self.valences_reasonable():
            # Try to reassign non-sigma bonds.
            self.reassign_nonsigma_bonds()
    # If was modified (say, atoms added/removed), some data becomes outdated.
    def changed(self):
        self.canonical_permutation=None
        self.inv_canonical_permutation=None
        self.colors=None
        # TO-DO: perhaps sort by classes in the beginning instead?
        self.nonequivalence_list=SortedList()
        self.equivalence_vector=np.repeat(-1, self.nhatoms())
        self.bond_equiv_dict={}
    # Checking graph's state.
    def valences_reasonable(self):
        for ha in self.hatoms:
            if not ha.valence_reasonable():
                return False
        return True
    def coordination_number(self, hatom_id):
        return len(self.neighbors(hatom_id))+self.hatoms[hatom_id].nhydrogens
    def atoms_equivalent(self, *atom_id_list):
        for i in range(len(atom_id_list)-1):
            if not self.atom_pair_equivalent(atom_id_list[i], atom_id_list[i+1]):
                return False
        return True
    def atom_pair_equivalent(self, atom_id1, atom_id2):
        if atom_id1==atom_id2:
            return True
        self.init_colors()
        if self.colors[atom_id1]!=self.colors[atom_id2]:
            return False
        if ((self.equivalence_vector[atom_id1]==-1) or (self.equivalence_vector[atom_id2]==-1)):
            atom_tuple=sorted_tuple(atom_id1, atom_id2)
            if atom_tuple in self.nonequivalence_list:
                return False
            else:
                output=self.uninit_atom_sets_equivalent([atom_id1], [atom_id2])
            if output:
                if ((self.equivalence_vector[atom_id1]==-1) and (self.equivalence_vector[atom_id2]==-1)):
                    new_class_id=max(self.equivalence_vector)+1
                    for i in [atom_id1, atom_id2]:
                        self.equivalence_vector[i]=new_class_id
                else:
                    if self.equivalence_vector[atom_id1]==-1:
                        self.equivalence_vector[atom_id1]=self.equivalence_vector[atom_id2]
                    else:
                        self.equivalence_vector[atom_id2]=self.equivalence_vector[atom_id1]
            else:
                self.nonequivalence_list.add(atom_tuple)
            return output
        else:
            return (self.equivalence_vector[atom_id1]==self.equivalence_vector[atom_id2])
    def uninit_atom_sets_equivalent(self, atom_set1, atom_set2):
        temp_colors1=copy.deepcopy(self.colors)
        temp_colors2=copy.deepcopy(self.colors)
        dummy_color=max(self.colors)+1
        for atom_id1, atom_id2 in zip(atom_set1, atom_set2):
            temp_colors1[atom_id1]=dummy_color
            temp_colors2[atom_id2]=dummy_color
        return self.graph.isomorphic_vf2(self.graph, color1=temp_colors1, color2=temp_colors2)
    #TO-DO a better graph-based way to do it?
    def pairs_equivalent(self, unsorted_tuple1, unsorted_tuple2):
        tuple1, tuple2=sorted_tuples(unsorted_tuple1, unsorted_tuple2)
        if tuple1==tuple2:
            return True
        if self.bond_order(*tuple1)!=self.bond_order(*tuple2):
            return False
        tuple2_matched=None
        for permutated_tuple2 in itertools.permutations(tuple2):
            ignore=False
            for atom1, atom2 in zip(tuple1, permutated_tuple2):
                if not self.atom_pair_equivalent(atom1, atom2):
                    ignore=True
                    break
            if not ignore:
                tuple2_matched=permutated_tuple2
                break
        if tuple2_matched is None:
            return False
        final_tuple=(tuple1, tuple2_matched)
        if final_tuple not in self.bond_equiv_dict:
            self.bond_equiv_dict[final_tuple]=self.uninit_atom_sets_equivalent(*final_tuple)
        return self.bond_equiv_dict[final_tuple]
    # How many times atom_id is repeated inside a molecule.
    def atom_multiplicity(self, atom_id):
        return sum(self.atom_pair_equivalent(atom_id, other_atom_id) for other_atom_id in range(self.nhatoms()))
    # Coordination number including unconnected electronic pairs. TO-DO: make sure it does not count pairs that contribute to an aromatic system?
    def effective_coordination_number(self, hatom_id):
        pairs=0
        hatom=self.hatoms[hatom_id]
        ncharge=hatom.ncharge
        if ncharge in unshared_pairs:
            cur_dict=unshared_pairs[ncharge]
            valence=hatom.valence
            if valence in cur_dict:
                pairs=cur_dict[valence]
        return self.coordination_number(hatom_id)+pairs
    # Hybridization of heavy atom hatom_id.
    def hybridization(self, hatom_id):
        return coord_num_hybrid[self.effective_coordination_number(hatom_id)]
    def is_connected(self):
        return (self.num_connected()==1)
    # Number of connected molecules.
    def num_connected(self):
        return len(self.graph.components())
    # Order of bond between atoms atom_id1 and atom_id2
    def bond_order(self, atom_id1, atom_id2):
        stuple=sorted_tuple(atom_id1, atom_id2)
        if stuple in self.bond_orders:
            return self.bond_orders[stuple]
        else:
            return 0
    # Number of heavy atoms.
    def nhatoms(self):
        return self.graph.vcount()
    # Total number of hydrogens.
    def tot_nhydrogens(self):
        return sum([hatom.nhydrogens for hatom in self.hatoms])
    # Dirty inheritance:
    def neighbors(self, hatom_id):
        return self.graph.neighbors(hatom_id)
    # Basic commands for managing the graph.
    def change_edge_order(self, atom1, atom2, change=0):
        if change != 0:
            if atom1==atom2:
                raise InvalidChange
            true_bond_tuple=tuple(sorted_tuple(atom1, atom2))
            try:
                bond_order=self.bond_orders[true_bond_tuple]
            except KeyError:
                bond_order=0
                self.graph.add_edge(*true_bond_tuple)
            bond_order+=change
            if bond_order<0:
                raise InvalidChange
            if bond_order==0:
                self.graph.delete_edges([true_bond_tuple])
                del(self.bond_orders[true_bond_tuple])
            else:
                self.bond_orders[true_bond_tuple]=bond_order

    def change_hydrogen_number(self, atom_id, hydrogen_number_change):
        self.hatoms[atom_id].nhydrogens+=hydrogen_number_change
        if self.hatoms[atom_id].nhydrogens<0:
            raise InvalidChange
    # For reassigning multiple bonds if valence composition is invalid, and all related procedures.
    def reassign_nonsigma_bonds(self):
        # Set all bond orders to one.
        for bond_tuple, bond_order in self.bond_orders.items():
            if bond_order>1:
                self.change_edge_order(*bond_tuple, 1-bond_order)
        # Find indices of atoms with spare non-sigma electrons. Also check coordination numbers are not above valence.
        coordination_numbers=[]
        extra_valence_indices=[]
        for hatom_id, hatom in enumerate(self.hatoms):
            cur_coord_number=self.coordination_number(hatom_id)
            max_valence=hatom.max_valence()
            if max_valence<cur_coord_number:
                raise InvalidAdjMat
            elif max_valence>cur_coord_number:
                coordination_numbers.append(cur_coord_number)
                extra_valence_indices.append(hatom_id)
            hatom.valence=hatom.smallest_valid_valence(cur_coord_number)
        extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list=self.extra_valence_subgraphs(extra_valence_indices, coordination_numbers)
        for extra_val_ids, coord_nums, extra_val_subgraph in zip(extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list):
            self.reassign_nonsigma_bonds_subgraph(extra_val_ids, coord_nums, extra_val_subgraph)

    def extra_valence_subgraphs(self, extra_valence_indices, coordination_numbers):
        total_subgraph=self.graph.induced_subgraph(extra_valence_indices)
        ts_components=total_subgraph.components()
        members=ts_components.membership
        extra_val_subgraph_list=ts_components.subgraphs()
        extra_val_ids_lists=sorted_by_membership(extra_valence_indices, members)
        coord_nums_lists=sorted_by_membership(coordination_numbers, members)
        return extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list

    def all_resonance_structures(self):
        # TO-DO: Maybe expand to include charges?
        extra_valence_indices=[]
        coordination_numbers=[]
        for hatom_id, hatom in enumerate(self.hatoms):
            cur_coord_number=self.coordination_number(hatom_id)
            cur_valence=hatom.valence
            if cur_valence>cur_coord_number:
                extra_valence_indices.append(hatom_id)
                coordination_numbers.append(cur_coord_number)
        if len(extra_valence_indices)==0:
            return []
        extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list=self.extra_valence_subgraphs(extra_valence_indices, coordination_numbers)
        output=[]
        for extra_val_ids, coord_nums, extra_val_subgraph in zip(extra_val_ids_lists, coord_nums_lists, extra_val_subgraph_list):
            added_edges_lists=self.complete_valences_attempt(extra_val_ids, coord_nums, extra_val_subgraph, all_possibilities=True)
            if added_edges_lists is None:
                raise InvalidAdjMat
            else:
                subgraph_res_struct=[]
                for added_edges in added_edges_lists:
                    add_bond_orders={}
                    for e in added_edges:
                        se=tuple(sorted(e))
                        if se in add_bond_orders:
                            add_bond_orders[se]+=1
                        else:
                            add_bond_orders[se]=1
                    if add_bond_orders not in subgraph_res_struct:
                        subgraph_res_struct.append(add_bond_orders)
                output.append(subgraph_res_struct)
        return output

    def reassign_nonsigma_bonds_subgraph(self, extra_val_ids, coord_nums, extra_val_subgraph, num_attempts=100000):
        added_edges=None
        HeavyAtomValenceIterators=[]
        IteratedValenceIds=[]
        for hatom_id, coord_num in zip(extra_val_ids, coord_nums):
            needed_val_id=self.hatoms[hatom_id].smallest_valid_valence(coord_num, True)
            if needed_val_id != -1:
                HeavyAtomValenceIterators.append(iter(self.hatoms[hatom_id].avail_val_list()[needed_val_id:]))
                IteratedValenceIds.append(hatom_id)
        HeavyAtomValences=list(itertools.product(*HeavyAtomValenceIterators))
        # We want the algorithm to return structure with smallest number of valence electrons possible, hence the sorting.
        HeavyAtomValences.sort(key=lambda x: sum(x))
        for HeavyAtomValencesList in HeavyAtomValences:
            # Assign all heavy atoms their current valences.
            for ha_id, ha_val in zip(IteratedValenceIds, HeavyAtomValencesList):
                self.hatoms[ha_id].valence=ha_val
            added_edges=self.complete_valences_attempt(extra_val_ids, coord_nums, extra_val_subgraph)
            if added_edges is not None:
                for added_edge in added_edges:
                    self.change_edge_order(*added_edge, 1)
                return
        raise InvalidAdjMat

    def complete_valences_attempt(self, extra_val_ids, coord_nums, extra_val_subgraph, all_possibilities=False):
        output=None
        added_edges=[]
        connection_opportunities=np.zeros(len(extra_val_ids), dtype=int)
        extra_valences=np.zeros(len(extra_val_ids), dtype=int)
        for i, (eval_id, coord_num) in enumerate(zip(extra_val_ids, coord_nums)):
            extra_valences[i]=self.hatoms[eval_id].valence-coord_num
        # TO-DO is it needed?
        if np.all(extra_valences==0):
            return []
        for cur_id, extra_valence in enumerate(extra_valences):
            if extra_valence!=0:
                neighs=extra_val_subgraph.neighbors(cur_id)
                for neigh in neighs:
                    if extra_valences[neigh] != 0:
                        connection_opportunities[cur_id]+=1
                if connection_opportunities[cur_id]==0:
                    return output
        saved_extra_valences={}
        saved_connection_opportunities={}
        saved_closed_atom={}
        saved_potential_other_atoms={}
        path_taken={}
        added_edges_stops={}
        cur_decision_fork=0
        while True:
            min_connectivity=0
            if np.any(connection_opportunities != 0):
                min_connectivity=np.min(connection_opportunities[np.nonzero(connection_opportunities)])
                closed_atom=np.where(connection_opportunities==min_connectivity)[0][0]
                potential_other_atoms=possible_closed_pairs(closed_atom, extra_valences, extra_val_subgraph)
            else:
                closed_atom=None
            if min_connectivity==1:
                choice=0
            else:
                cur_decision_fork+=1
                if closed_atom is None:
                    if (np.any(extra_valences!=0) or all_possibilities):
                        # Fall back to an earlier save point
                        while True:
                            cur_decision_fork-=1
                            if cur_decision_fork==0:
                                return output
                            path_taken[cur_decision_fork]+=1
                            if path_taken[cur_decision_fork] != len(saved_potential_other_atoms[cur_decision_fork]):
                                break
                    extra_valences[:]=saved_extra_valences[cur_decision_fork][:]
                    connection_opportunities[:]=saved_connection_opportunities[cur_decision_fork][:]
                    potential_other_atoms=copy.deepcopy(saved_potential_other_atoms[cur_decision_fork])
                    closed_atom=saved_closed_atom[cur_decision_fork]
                    del(added_edges[added_edges_stops[cur_decision_fork]:])
                else:
                    path_taken[cur_decision_fork]=0
                    saved_extra_valences[cur_decision_fork]=np.copy(extra_valences)
                    saved_connection_opportunities[cur_decision_fork]=np.copy(connection_opportunities)
                    saved_potential_other_atoms[cur_decision_fork]=copy.deepcopy(potential_other_atoms)
                    saved_closed_atom[cur_decision_fork]=closed_atom
                    added_edges_stops[cur_decision_fork]=len(added_edges)
                choice=path_taken[cur_decision_fork]
            other_closed_atom=potential_other_atoms[choice]

            added_edges.append((extra_val_ids[closed_atom], extra_val_ids[other_closed_atom]))
            for cur_id in [closed_atom, other_closed_atom]:
                extra_valences[cur_id]-=1
                if extra_valences[cur_id]==0:
                    connection_opportunities[cur_id]=0
                    for neigh_id in extra_val_subgraph.neighbors(cur_id):
                        if connection_opportunities[neigh_id]!=0:
                            connection_opportunities[neigh_id]-=1
            if np.all(extra_valences==0):
                if all_possibilities:
                    if output is None:
                        output=[copy.deepcopy(added_edges)]
                    else:
                        output.append(copy.deepcopy(added_edges))
                else:
                    return added_edges
        
    # More sophisticated commands that are to be called in the "modify" module.
    def change_bond_order(self, atom1, atom2, bond_order_change):
        if bond_order_change!=0:
            self.change_edge_order(atom1, atom2, bond_order_change)
            for atom_id in [atom1, atom2]:
                self.change_hydrogen_number(atom_id, -bond_order_change)
        self.changed()
    def remove_heavy_atom(self, atom_id):
        for neigh_id in self.neighbors(atom_id):
            cur_bond_order=self.bond_order(neigh_id, atom_id)
            self.change_bond_order(atom_id, neigh_id, -cur_bond_order)
        self.graph.delete_vertices([atom_id])
        #TO-DO: is it possible to instead tie dict keys to edges of self.graph?
        new_bond_orders={}
        for bond_tuple, bond_order in self.bond_orders.items():
            new_tuple=[]
            for b in bond_tuple:
                if b>atom_id:
                    b-=1
                new_tuple.append(b)
            new_bond_orders[tuple(new_tuple)]=bond_order
        self.bond_orders=new_bond_orders
        del(self.hatoms[atom_id])
        self.changed()
    def add_heavy_atom_chain(self, modified_atom_id, new_chain_atoms, new_chain_atom_valences=None):
        bonded_chain=[modified_atom_id]
        last_added_id=self.nhatoms()
        num_added_atoms=len(new_chain_atoms)
        self.graph.add_vertices(num_added_atoms)
        for new_chain_atom in new_chain_atoms:
            bonded_chain.append(last_added_id)
            last_added_id+=1
            self.hatoms.append(HeavyAtom(new_chain_atom))
            self.hatoms[-1].nhydrogens=self.hatoms[-1].valence
        for i in range(num_added_atoms):
            self.change_bond_order(bonded_chain[i], bonded_chain[i+1], 1)
        self.changed()
    def replace_heavy_atom(self, replaced_atom_id, inserted_atom, inserted_valence=None):
        # Extracting full chemical information of the graph.
        self.hatoms[replaced_atom_id].ncharge=int_atom_checked(inserted_atom)
        old_valence=self.hatoms[replaced_atom_id].valence
        if inserted_valence is None:
            inserted_valence=default_valence(inserted_atom)
        self.hatoms[replaced_atom_id].valence=inserted_valence
        self.change_hydrogen_number(replaced_atom_id, inserted_valence-old_valence)
        self.changed()
    def change_valence(self, modified_atom_id, new_valence):
        self.change_hydrogen_number(modified_atom_id, new_valence-self.hatoms[modified_atom_id].valence)
        self.hatoms[modified_atom_id].valence=new_valence
        self.changed()
    # Output properties that include hydrogens.
    def full_natoms(self):
        return sum([ha.nhydrogens for ha in self.hatoms])+self.nhatoms()
    def full_ncharges(self):
        output=np.ones(self.full_natoms(), dtype=int)
        for ha_id, ha in enumerate(self.hatoms):
            output[ha_id]=ha.ncharge
        return output
    def full_adjmat(self):
        natoms=self.full_natoms()
        output=np.zeros((natoms, natoms), dtype=int)
        for bond_tuple, bond_order in self.bond_orders.items():
            output[bond_tuple]=bond_order
            output[bond_tuple[::-1]]=bond_order
        cur_h_id=self.nhatoms()
        for ha_id, ha in enumerate(self.hatoms):
            for h_counter in range(ha.nhydrogens):
                output[ha_id, cur_h_id]=1
                output[cur_h_id, ha_id]=1
                cur_h_id+=1
        return output
    # For properties used to generate representations.
    def get_res_av_bond_orders(self, edge_list=None):
        if edge_list is None:
            edge_list=self.graph.get_edgelist()
        # Get dictionnaries for all resonance structures.
        res_struct_dict_list=self.all_resonance_structures()
        res_av_bos=np.ones(len(edge_list), dtype=float)
        all_av_res_struct_dict={}
        for res_struct_dict in res_struct_dict_list:
            av_res_struct_dict=res_struct_dict[0]
            for add_rsd in res_struct_dict[1:]:
                for bond_edge, bond_order in add_rsd.items():
                    if bond_edge in av_res_struct_dict:
                        av_res_struct_dict[bond_edge]+=bond_order
                    else:
                        av_res_struct_dict[bond_edge]=bond_order
            for edge in av_res_struct_dict:
                av_res_struct_dict[edge]=float(av_res_struct_dict[edge])/len(res_struct_dict)
            all_av_res_struct_dict={**all_av_res_struct_dict, **av_res_struct_dict}
        for eid, edge in enumerate(edge_list):
            if edge in all_av_res_struct_dict:
                res_av_bos[eid]+=all_av_res_struct_dict[edge]
        return res_av_bos
    def get_res_av_bond_lengths(self, edge_list=None):
        if edge_list is None:
            edge_list=self.graph.get_edgelist()
        # Get weighted bond orders.
        res_av_bos=self.get_res_av_bond_orders(edge_list=edge_list)
        bond_lengths=np.zeros(len(edge_list), dtype=float)
        for eid, (e, bo) in enumerate(zip(edge_list, res_av_bos)):
            bo2 = int(bo+irrelevant_bond_order_difference)
            bo1 = bo2-1
            bo2_frac=bo-bo2
            bo1_frac=1.0-bo2_frac
            els=[periodic_table[self.hatoms[a].ncharge] for a in e]
            for int_bo, int_bo_frac in [(bo2, bo2_frac), (bo1, bo1_frac)]:
                for el in els:
                    ar_list=atom_radii[el]
                    if len(ar_list)>int_bo:
                        bond_lengths[eid]+=int_bo_frac*ar_list[int_bo]
        return bond_lengths
    def shortest_paths(self, weights=None):
        if self.nhatoms()==1:
            return np.array([[0.0]])
        else:
            return np.array(self.graph.shortest_paths(weights=weights))
    # Procedures used for sorting.
    def init_colors(self):
        if self.colors is None:
            self.colors=list2colors(self.hatoms)
    def init_canonical_permutation(self):
        if self.canonical_permutation is None:
            self.init_colors()
            self.canonical_permutation=self.nhatoms()-1-np.array(self.graph.canonical_permutation(color=self.colors))
            self.inv_canonical_permutation=np.zeros(self.nhatoms(), dtype=int)
            for pos_counter, pos in enumerate(self.canonical_permutation):
                self.inv_canonical_permutation[pos]=pos_counter
    def get_inv_canonical_permutation(self):
        self.init_canonical_permutation()
        return self.inv_canonical_permutation
    def comparison_iterator(self):
        iterators=[iter([self.nhatoms()])]
        if self.canonical_permutation is None:
            self.init_canonical_permutation()
        iterators.append([self.hatoms[hatom_id] for hatom_id in self.inv_canonical_permutation])
        for hatom_id in self.inv_canonical_permutation:
            neighbor_permuted_list=[self.canonical_permutation[neigh_id] for neigh_id in self.neighbors(hatom_id)]
            iterators.append(iter([len(neighbor_permuted_list)]))
            iterators.append(iter(sorted(neighbor_permuted_list)))
        return itertools.chain(*iterators)
    def __lt__(self, ch2):
        return (triple_gt_witer(self, ch2) is False)
    def __gt__(self, ch2):
        return (triple_gt_witer(self, ch2) is True)
    def __eq__(self, ch2):
        return (triple_gt_witer(self, ch2) is None)
    def __str__(self):
        return "heavy atoms: "+str(self.hatoms)+" , bonds&orders: "+str(self.bond_orders)
    def __repr__(self):
        return str(self)


# For merging chemgraphs together or splitting them.
def combine_chemgraphs(cg1, cg2, connection_tuples=[], count_equivalences=False):
    new_graph=disjoint_union([cg1.graph, cg2.graph])
    new_hatoms=copy.deepcopy(cg1.hatoms+cg2.hatoms)
    id2_shift=cg1.nhatoms()
    new_bond_orders=copy.deepcopy(cg1.bond_orders)
    for bond_tuple, bond_order in cg2.bond_orders.items():
        new_bond_tuple=tuple(np.array(bond_tuple)+id2_shift)
        new_bond_orders[new_bond_tuple]=bond_order
    output=ChemGraph(graph=new_graph, hatoms=new_hatoms, bond_orders=new_bond_orders)
    if count_equivalences:
        equiv_count=1
    for connection_tuple in connection_tuples:
        connection_args=list(connection_tuple)
        connection_args[1]+=id2_shift
        if count_equivalences:
            num_opportunities=0
            for id1 in range(output.nhatoms()):
                for id2 in range(id1):
                    if output.pairs_equivalent((id1, id2), connection_args):
                        num_opportunities+=1
            equiv_count*=num_opportunities
        output.change_bond_order(*connection_args)
    if count_equivalences:
        return output, equiv_count
    else:
        return output

#TO-DO this partially duplicates with splitting EGC's in utility.py
def split_chemgraph_no_dissociation_check(cg_input, membership_vector, copied=False):
    if copied:
        cg=cg_input
    else:
        cg=copy.deepcopy(cg_input)
    subgraphs_vertex_ids=sorted_by_membership(list(range(cg.nhatoms())), membership_vector)
    new_graph_list=[]
    new_hatoms_list=[]
    new_bond_orders_list=[]
    new_bond_positions_orders_dict_list=[]
    for vertex_ids in subgraphs_vertex_ids:
        new_graph_list.append(cg.graph.subgraph(vertex_ids))
        new_hatoms_list.append([cg.hatoms[vertex_id] for vertex_id in vertex_ids])
        new_bond_orders_list.append({})
        new_bond_positions_orders_dict_list.append({})
    # Create bond order dictionnaries:
    for vertex_id in range(cg.nhatoms()):
        mv1=membership_vector[vertex_id]
        internal_id1=subgraphs_vertex_ids[mv1].index(vertex_id)
        for neigh in cg.graph.neighbors(vertex_id):
            mv2=membership_vector[neigh]
            internal_id2=subgraphs_vertex_ids[mv2].index(neigh)
            cur_bond_order=cg.bond_order(vertex_id, neigh)
            if mv1==mv2:
                new_bond_orders_list[mv1]={**new_bond_orders_list[mv1], sorted_tuple(internal_id1, internal_id2) : cur_bond_order}
            else:
                if internal_id1 in new_bond_positions_orders_dict_list[mv1]:
                    new_bond_positions_orders_dict_list[mv1][internal_id1].append(cur_bond_order)
                else:
                    new_bond_positions_orders_dict_list[mv1][internal_id1]=[cur_bond_order]
    output=[]
    for graph, hatoms, bond_orders, bond_positions_orders_dict in zip(new_graph_list, new_hatoms_list, new_bond_orders_list, new_bond_positions_orders_dict_list):
        bond_positions_orders=[]
        for atom_id, connection_orders in bond_positions_orders_dict.items():
            for connection_order in connection_orders:
                bond_positions_orders.append((atom_id, connection_order))
                hatoms[atom_id].nhydrogens+=connection_order
        new_cg=ChemGraph(graph=graph, hatoms=hatoms, bond_orders=bond_orders)
        output.append(Fragment(chemgraph=new_cg, bond_positions_orders=bond_positions_orders))
    return output

def split_chemgraph(cg, membership_vector):
    output=[]
    cg_copy=copy.deepcopy(cg)
    fragments=split_chemgraph_no_dissociation_check(cg_copy, membership_vector, copied=True)
    for fragment in fragments:
        if fragment.chemgraph.num_connected()==1:
            output.append(fragment)
        else:
            # Break it further down.
            fragment_mv=np.array(fragment.chemgraph.graph.components().membership)
            connection_opportunities=np.zeros(len(fragment_mv), dtype=int)
            for (bp, bo) in fragment.bond_positions_orders:
                connection_opportunities[bp]=bo
            sorted_connection_opportunities=sorted_by_membership(connection_opportunities, fragment_mv)
            additional_frags=split_chemgraph_no_dissociation_check(fragment.chemgraph, fragment_mv)
            new_fragments=[]
            for f, cur_connection_opportunities in zip(additional_frags, sorted_connection_opportunities):
                for co_id, co in enumerate(cur_connection_opportunities):
                    if co != 0:
                        f.bond_positions_orders.append((co_id, co))
                new_fragments.append(f)
            new_fragments.sort(key = lambda x : x.chemgraph.nhatoms(), reverse=True)
            output+=new_fragments
    return output




def connection_forbidden(nc1, nc2, forbidden_bonds):
    if ((nc1 is None) or (nc2 is None) or (forbidden_bonds is None)):
        return False
    nc_tuple=sorted_tuple(int_atom_checked(nc1), int_atom_checked(nc2))
    return (nc_tuple in forbidden_bonds)

class Fragment:
    def __init__(self, chemgraph=None, bond_positions_orders=[], **chemgraph_kwargs):
        if chemgraph is None:
            self.chemgraph=ChemGraph(**chemgraph_kwargs)
        else:
            self.chemgraph=chemgraph
        if len(bond_positions_orders)==0:
            self.bond_positions_orders=[]
        else:
            self.bond_positions_orders=sorted(bond_positions_orders, key = lambda x : x[::-1])
            # TO-DO Does it make sense to forbid position duplication?
#            for i, t in enumerate(self.bond_positions_orders[:-1]):
#                if t[0]==self.bond_positions_orders[i+1][0]:
#                    raise Exception("Duplicated bonding position.")
    def add_to(self, chemgraph, connecting_positions):
        connection_tuples=[(conn_pos, *bond_pos_order) for conn_pos, bond_pos_order in zip(connecting_positions, self.bond_positions_orders)]
        return combine_chemgraphs(chemgraph, self.chemgraph, connection_tuples)
    #TO-DO revise for symmetry's sake? Or perhaps add everything related to detailed balance inside genetic algorithms?
    def connection_opportunities(self, chemgraph, forbidden_bonds=None):
        output=[]
        for checked_hatom_ids in itertools.product(*[range(chemgraph.nhatoms()) for i in range(len(self.bond_positions_orders))]):
            to_add=True
            extra_hydrogens={}
            for checked_hatom_id, (other_atom_id, needed_extra_valence) in zip(checked_hatom_ids, self.bond_positions_orders):
                if connection_forbidden(self.chemgraph.hatoms[other_atom_id].ncharge, chemgraph.hatoms[checked_hatom_id].ncharge, forbidden_bonds):
                    to_add=False
                    break
                if checked_hatom_id not in extra_hydrogens:
                    extra_hydrogens[checked_hatom_id]=chemgraph.hatoms[checked_hatom_id].nhydrogens
                extra_hydrogens[checked_hatom_id]-=needed_extra_valence
                if extra_hydrogens[checked_hatom_id]<0:
                    to_add=False
                    break
            if to_add:
                output.append(checked_hatom_ids)
        return output
    def all_connections_with_frag(self, other_fragment, forbidden_bonds=None):
        if self.bo_list() != other_fragment.bo_list():
            return []
        permutation_iterators=[]
        cur_bo=-1
        cur_iterator=None
        for bp, bo in other_fragment.bond_positions_orders:
            if bo==cur_bo:
                cur_iterator.append(bp)
            else:
                cur_bo=bo
                if cur_iterator is not None:
                    permutation_iterators.append(itertools.permutations(cur_iterator))
                cur_iterator=[bp]
        permutation_iterators.append(itertools.permutations(cur_iterator))

        conn_opp_candidates=[]
        for perm_lists in itertools.product(*permutation_iterators):
            new_candidate=[]
            for pl in perm_lists:
                new_candidate+=list(pl)
            conn_opp_candidates.append(new_candidate)
        if forbidden_bonds is not None:
            filtered_conn_opp_candidates=[]
            for conn_opp_candidate in conn_opp_candidates:
                to_add=True
                for (conn_atom_self, bo), conn_atom_other in zip(self.bond_positions_orders, conn_opp_candidate):
                    if connection_forbidden(self.chemgraph.hatoms[conn_atom_self].ncharge, other_fragment.chemgraph.hatoms[conn_atom_other].ncharge, forbidden_bonds):
                        to_add=False
                        break
                if to_add:
                    filtered_conn_opp_candidates.append(conn_opp_candidate)
            conn_opp_candidates=filtered_conn_opp_candidates
        output=SortedList()
        for conn_opp_candidate in conn_opp_candidates:
            new_mol=self.add_to(other_fragment.chemgraph, conn_opp_candidate)
            if new_mol not in output:
                output.add(new_mol)
        return output

    def bo_list(self):
        return [bo for (bp, bo) in self.bond_positions_orders]
    def __repr__(self):
        return str(self)
    def __str__(self):
        output=str(self.chemgraph)+"\nConnection positions: "
        for bp, bo in self.bond_positions_orders:
            output+="("+str(bp)+","+str(bo)+")"
        return output
    def comparison_iterator(self):
        cg_comp_iter=self.chemgraph.comparison_iterator()
        cn_bond_positions_orders=[(self.chemgraph.canonical_permutation[bp], bo) for (bp, bo) in self.bond_positions_orders]
        cn_bond_positions_orders.sort(key = lambda x : x[0])
        return itertools.chain(cg_comp_iter, iter(cn_bond_positions_orders))
    def __lt__(self, ch2):
        return (triple_gt_witer(self, ch2) is False)
    def __gt__(self, ch2):
        return (triple_gt_witer(self, ch2) is True)
    def __eq__(self, ch2):
        return (triple_gt_witer(self, ch2) is None)

#   Utility functions
def possible_closed_pairs(closed_atom, extra_valences, extra_val_subgraph):
    output=[]
    for i in extra_val_subgraph.neighbors(closed_atom):
        if extra_valences[i]!=0:
            output.append(i)
    return output
