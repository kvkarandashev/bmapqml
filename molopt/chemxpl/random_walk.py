# TO-DO revise everything associated with nhatoms range?
from sortedcontainers import SortedList
from .ext_graph_compound import ExtGraphCompound
from .modify import atom_replacement_possibilities, atom_removal_possibilities, chain_addition_possibilities, bond_change_possibilities, valence_change_possibilities,\
                        add_heavy_atom_chain, remove_heavy_atom, replace_heavy_atom, change_bond_order, change_valence, randomized_cross_coupling
from .utils import rdkit_to_egc, egc_to_rdkit
from .valence_treatment import sorted_tuple, connection_forbidden
from .periodic import element_name
import random, copy
import numpy as np
try:
    from xyz2mol import str_atom
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install xyz2mol software in order to use this parser. '
                      'Visit: https://github.com/jensengroup/xyz2mol')


# TO-DO 1. make default values for randomized change parameters work. 2. Add atoms with bond order more than one already?

default_change_list=[add_heavy_atom_chain, remove_heavy_atom, replace_heavy_atom, change_bond_order, change_valence]

stochiometry_conserving_change_list=[change_bond_order, change_valence]

inverse_procedure={add_heavy_atom_chain : remove_heavy_atom, remove_heavy_atom : add_heavy_atom_chain,
                replace_heavy_atom : replace_heavy_atom, change_bond_order : change_bond_order, change_valence : change_valence}

change_possibility_label={add_heavy_atom_chain : "possible_elements", remove_heavy_atom : "possible_elements",
                replace_heavy_atom : "possible_elements", change_bond_order : "bond_order_changes", change_valence : None}

def inverse_possibility_label(change_function, possibility_label):
    if change_function is change_bond_order:
        return -possibility_label
    return possibility_label

def egc_change_func(egc_in, possibility_label, final_possibility_val, change_function):
    if change_function is change_bond_order:
        return change_function(egc_in, *final_possibility_val, possibility_label)
    if change_function is remove_heavy_atom:
        return change_function(egc_in, final_possibility_val)
    if change_function is change_valence:
        return change_function(egc_in, possibility_label, final_possibility_val)
    true_possibility_label=possibility_label
    if change_function is add_heavy_atom_chain:
        true_possibility_label=[possibility_label]
    return change_function(egc_in, final_possibility_val, true_possibility_label)

def available_options_prob_norm(dict_in):
    output=0.0
    for i in list(dict_in):
        if len(dict_in[i])!=0:
            output+=1.0
    return output

def random_choice_from_dict(possibilities, choices=None, get_probability_of=None):
    prob_sum=0.0
    corr_prob_choice={}
    if choices is None:
        choices=list(possibilities.keys())
    for choice in list(choices):
        if ((choice not in list(possibilities)) or (len(possibilities[choice])==0)):
            continue
        if isinstance(choices, dict):
            prob=choices[choice]
        else:
            prob=1.0
        prob_sum+=prob
        corr_prob_choice[choice]=prob
    if get_probability_of is None:
        final_choice=random.choices(list(corr_prob_choice.keys()), list(corr_prob_choice.values()))[0]
        final_prob_log=np.log(corr_prob_choice[final_choice]/prob_sum)
        return final_choice, possibilities[final_choice], final_prob_log
    else:
        return possibilities[get_probability_of], np.log(corr_prob_choice[get_probability_of]/prob_sum)

def lookup_or_none(dict_in, key):
    if key in dict_in:
        return dict_in[key]
    else:
        return None

def randomized_change(tp, change_prob_dict=default_change_list, **other_kwargs):
    cur_change_procedure, possibilities, total_forward_prob=random_choice_from_dict(tp.possibilities(), change_prob_dict)
    possibility_dict_label=change_possibility_label[cur_change_procedure]
    possibility_dict=lookup_or_none(other_kwargs, possibility_dict_label)
    possibility_label, change_possibilities, forward_prob=random_choice_from_dict(possibilities, possibility_dict)
    total_forward_prob+=forward_prob-np.log(len(change_possibilities))
    final_possibility_val=random.choice(change_possibilities)

    new_egc=egc_change_func(tp.egc, possibility_label, final_possibility_val, cur_change_procedure)

    new_tp=TrajectoryPoint(egc=new_egc)
    new_tp.init_possibility_info(**other_kwargs)
    # Calculate the chances of doing the inverse operation
    inv_proc=inverse_procedure[cur_change_procedure]
    inverse_possibilities, total_inverse_prob=random_choice_from_dict(new_tp.possibilities(), change_prob_dict, get_probability_of=inv_proc)
    if cur_change_procedure is change_valence:
        inverse_prob=-np.log(len(new_tp.possibilities()[cur_change_procedure]))
        inverse_final_possibilities=change_possibilities
    else:
        inv_poss_dict_label=change_possibility_label[inv_proc]
        inv_poss_dict=lookup_or_none(other_kwargs, inv_poss_dict_label)
        if cur_change_procedure is replace_heavy_atom:
            inverse_pl=element_name[tp.egc.chemgraph.hatoms[final_possibility_val].ncharge]
        else:
            inverse_pl=inverse_possibility_label(cur_change_procedure, possibility_label)
        inverse_final_possibilities, inverse_prob=random_choice_from_dict(inverse_possibilities, inv_poss_dict,
                                get_probability_of=inverse_pl)
    total_inverse_prob+=inverse_prob-np.log(len(inverse_final_possibilities))
    
    return new_tp, total_forward_prob-total_inverse_prob

# This class stores all the infromation needed to preserve detailed balance of the random walk.
class TrajectoryPoint:
    def __init__(self, egc=None, cg=None, num_visits=0):
        if egc is None:
            if cg is not None:
                egc=ExtGraphCompound(chemgraph=cg)
        self.egc=egc
        self.num_visits=num_visits
        # Information for keeping detailed balance.
        self.bond_order_change_possibilities=None
        self.nuclear_charge_change_possibilities=None
        self.atom_removal_possibilities=None
        self.valence_change_possibilities=None
        self.chain_addition_possibilities=None

        self.additional_data=None
    # TO-DO better way to write this?
    def init_possibility_info(self, bond_order_changes=[-1, 1], possible_elements=['C'], **other_kwargs):
        if self.bond_order_change_possibilities is None:
            self.bond_order_change_possibilities={}
            for bond_order_change in bond_order_changes:
                cur_possibilities=bond_change_possibilities(self.egc, bond_order_change, **other_kwargs)
                if len(cur_possibilities) != 0:
                    self.bond_order_change_possibilities[bond_order_change]=cur_possibilities
            self.nuclear_charge_change_possibilities={}
            self.atom_removal_possibilities={}
            self.chain_addition_possibilities={}
            for possible_element in possible_elements:
                repl_possibilities=atom_replacement_possibilities(self.egc, possible_element, **other_kwargs)
                if len(repl_possibilities) != 0:
                    self.nuclear_charge_change_possibilities[possible_element]=repl_possibilities
                removal_possibilities=atom_removal_possibilities(self.egc, deleted_atom=possible_element, only_end_atoms=True, **other_kwargs)
                if len(removal_possibilities) != 0:
                    self.atom_removal_possibilities[possible_element]=removal_possibilities
                addition_possibilities=chain_addition_possibilities(self.egc, chain_starting_element=possible_element, **other_kwargs)
                if len(addition_possibilities) != 0:
                    self.chain_addition_possibilities[possible_element]=addition_possibilities
            self.valence_change_possibilities=valence_change_possibilities(self.egc)
    def possibilities(self):
        return {add_heavy_atom_chain : self.chain_addition_possibilities,
                    replace_heavy_atom : self.nuclear_charge_change_possibilities,
                    remove_heavy_atom : self.atom_removal_possibilities,
                    change_bond_order : self.bond_order_change_possibilities,
                    change_valence : self.valence_change_possibilities}
    def update_additional_data(self, additional_data_func):
        new_add_data=additional_data_func(self)
        for label, val in new_add_data.items():
            self.additional_data[label]=val
    def calc_min_function(self, min_function):
        return min_function(**self.additional_data)
    def __lt__(self, tp2):
        return (self.egc < tp2.egc)
    def __gt__(self, tp2):
        return (self.egc > tp2.egc)
    def __eq__(self, tp2):
        return (self.egc == tp2.egc)
    def __str__(self):
        return str(self.egc)
    def __repr__(self):
        return str(self)

# Rules for accepting trial steps.
def Metropolis_acceptance(delta_pot):
    if delta_pot<=0.0:
        return True
    else:
        return (random.random() < np.exp(-delta_pot))

def greedy_acceptance(delta_pot):
    return (delta_pot<=0.0)

acceptance_rules={"Metropolis_acceptance" : Metropolis_acceptance, "greedy_acceptance" : greedy_acceptance}

def tidy_forbidden_bonds(forbidden_bonds):
    if forbidden_bonds is None:
        return None
    output=SortedList()
    for fb in forbidden_bonds:
        output.add(sorted_tuple(*fb))
    return output

def has_forbidden_bonds(egc, forbidden_bonds=None):
    adjmat=egc.true_adjmat()
    for i1, nc1 in enumerate(egc.nuclear_charges[:egc.num_heavy_atoms()]):
        for i2, nc2 in enumerate(egc.nuclear_charges[:i1]):
            if adjmat[i1, i2] != 0:
                if connection_forbidden(nc1, nc2, forbidden_bonds):
                    return True

mol_egc_converter={"rdkit" : (rdkit_to_egc, egc_to_rdkit)}

def conv_func_mol_egc(inp, forward=True, mol_format=None):
    if mol_format is None:
        return inp
    else:
        if forward:
            conv_func_id=0
        else:
            conv_func_id=1
        return mol_egc_converter[mol_format][conv_func_id](inp)

# TO-DO replace biasing potential with histogram.

# Class that generates a trajectory over chemical space.
class RandomWalk:
    def __init__(self, init_egcs=None, bias_coeff=None, randomized_change_params={}, acceptance_rule="Metropolis_acceptance", restart_trajectory=None,
                    conserve_stochiometry=False, bound_enforcing_coeff=1.0, keep_histogram=False, beta=None, min_function=None, additional_data_function=None, num_replicas=None):
        self.num_replicas=num_replicas
        if isinstance(beta, list):
            self.num_replicas=len(beta)
        if isinstance(init_egcs, list):
            self.num_replicas=len(init_egcs)
        else:
            if self.num_replicas is None:
                self.num_replicas=1
            init_egcs=[init_egcs for i in range(self.num_replicas)]
        self.beta=beta
        if isinstance(self.beta, list):
            assert(len(self.beta)==len(self.num_replicas))
        else:
            if self.beta is not None:
                self.beta=[self.beta for i in range(self.num_replicas)]

        self.cur_tps=[TrajectoryPoint(egc=init_egc) for init_egc in init_egcs]
        if restart_trajectory is None:
            self.explored_egcs=SortedList()
        else:
            self.explored_egcs=restart_trajectory
        self.bias_coeff=bias_coeff
        self.randomized_change_params=randomized_change_params
        self.init_randomized_change_params(randomized_change_params)
        self.acceptance_rule=acceptance_rules[acceptance_rule]
        self.keep_histogram=keep_histogram
        if self.keep_histogram:
            self.update_histogram()
        self.change_list=default_change_list
        self.bound_enforcing_coeff=bound_enforcing_coeff
        self.conserve_stochiometry=conserve_stochiometry
        if self.conserve_stochiometry is True:
            self.change_list=stochiometry_conserving_change_list

        self.hydrogen_num=None

        self.beta=beta
        self.min_function=min_function
        self.additional_data_function=additional_data_function

        if self.additional_data_function is not None:
            for i in range(self.num_replicas):
                self.cur_tps[i].update_additional_data(self.additional_data_function)

        # For storing statistics on move success.
        self.num_attempted_cross_couplings=0
        self.num_accepted_cross_couplings=0

    def init_randomized_change_params(self, randomized_change_params=None):
        if randomized_change_params is not None:
            self.randomized_change_params=randomized_change_params
            if "forbidden_bonds" in self.randomized_change_params:
                self.randomized_change_params["forbidden_bonds"]=tidy_forbidden_bonds(self.randomized_change_params["forbidden_bonds"])
        self.used_randomized_change_params=copy.deepcopy(self.randomized_change_params)

    def accept_reject_move(self, new_tps, prob_balance, replica_ids=[0]):
        prev_tps=[self.cur_tps[replica_id] for replica_id in replica_ids]
        delta_pot=prob_balance
        for new_tp, prev_tp in zip(new_tps, prev_tps):
            if self.min_function is not None:
                prob_balance+=self.beta[replica_id]*(new_tp.calc_min_function(self.min_function)-change_tp.calc_min_function(self.min_function))
            if self.bias_coeff is not None:
                delta_pot+=self.biasing_potential(new_tp)-self.biasing_potential(prev_tp)
            if self.bound_enforcing_coeff is not None:
                delta_pot+=self.bound_enforcing_pot(new_tp)-self.bound_enforcing_pot(prev_tp)

        accepted=self.acceptance_rule(delta_pot)

        if accepted:
            for new_tp_id, replica_id in enumerate(replica_ids):
                self.cur_tps[replica_id]=new_tps[new_tp_id]
        return accepted

    def MC_step(self, show_acceptance=False, change_histogram=True, replica_id=0):
        changed_tp=self.cur_tps[replica_id]
        changed_tp.init_possibility_info(**self.used_randomized_change_params)
        new_tp, prob_balance=randomized_change(changed_tp, change_prob_dict=self.change_list, **self.used_randomized_change_params)
        accepted=self.accept_reject_move([new_tp], prob_balance, replica_ids=[replica_id])
        if (self.keep_histogram and change_histogram):
            self.update_histogram()
        if show_acceptance:
            return self.cur_tps[replica_id], accepted
        else:
            return self.cur_tps[replica_id]

    def MC_step_all(self, **mc_step_kwargs):
        output=[]
        for replica_id in range(self.num_replicas):
            output.append(self.MC_step(**mc_step_kwargs, replica_id=replica_id))
        return output

    def bound_enforcing_pot(self, tp):
        return self.bound_enforcing_coeff*self.bound_outlie(tp.egc)

    def bound_outlie(self, egc):
        output=0
        if self.hydrogen_num is not None:
            if egc.chemgraph.tot_nhydrogens() != self.hydrogen_num:
                output+=abs(egc.chemgraph.tot_nhydrogens()-self.hydrogen_num)
        if egc.chemgraph.num_connected != 1:
            output+=egc.chemgraph.num_connected()-1
        if "final_nhatoms_range" in self.randomized_change_params:
            final_nhatoms_range=self.randomized_change_params["final_nhatoms_range"]
            if egc.num_heavy_atoms() > final_nhatoms_range[1]:
                output+=egc.num_heavy_atoms()-final_nhatoms_range[1]
            else:
                if egc.num_heavy_atoms() < final_nhatoms_range[0]:
                    output+=final_nhatoms_range[0]-egc.num_heavy_atoms()
        return output
    # TO-DO: 1. Currently works in a way that prohibits a fragmented molecule to be returned unless it's stiched together.
    def change_molecule(self, mol_in, randomized_change_params=None, skip_unaccepted=False, mol_format=None):
        init_egc=conv_func_mol_egc(mol_in, mol_format=mol_format)
        if self.conserve_stochiometry:
            self.hydrogen_num=init_egc.chemgraph.tot_nhydrogens()
        self.init_randomized_change_params(randomized_change_params=randomized_change_params)
        if "forbidden_bonds" in self.randomized_change_params:
            if has_forbidden_bonds(init_egc, self.randomized_change_params["forbidden_bonds"]):
                raise Exception("Forbidden bond in initial compound:"+str(init_egc))
        self.cur_tps=[TrajectoryPoint(egc=init_egc)]
        # TO-DO: That's an argument for making sure initializing randomized_change_params once and for all in __init__ is better.
        move_accepted=False
        while not move_accepted:
            new_tp, move_accepted=self.MC_step(show_acceptance=True)
#           TO-DO Is it bad for detailed balance.
            if move_accepted:
                move_accepted=(self.bound_outlie(new_tp.egc)==0)
            if not skip_unaccepted:
                move_accepted=True
        return conv_func_mol_egc(new_tp.egc, mol_format=mol_format, forward=False)
    def change_molecule_list(self, mol_in_list, **change_molecule_kwargs):
        output=[]
        for mol in mol_in_list:
            output.append(self.change_molecule(mol, **change_molecule_kwargs))
        return output
    def genetic_MC_step(self, num_attempted_changes=1, randomized_change_params=None):
        self.init_randomized_change_params(randomized_change_params=randomized_change_params)
        for attempted_change_counter in range(num_attempted_changes):
            old_ids=np.random.randint(self.num_replicas, size=(2,))
            old_cg_pair=[self.cur_tps[old_id].egc.chemgraph for old_id in old_ids]
            new_cg_pair, prob_balance=randomized_cross_coupling(old_cg_pair, **self.used_randomized_change_params)
            accepted=False
            self.num_attempted_cross_couplings+=1
            if new_cg_pair is not None:
                if "nhatoms_range" in self.used_randomized_change_params:
                    nhatoms_range=self.used_randomized_change_params["nhatoms_range"]
                    invalid=False
                    for nm in new_cg_pair:
                        invalid=((nm.nhatoms()<nhatoms_range[0]) or (nm.nhatoms()>nhatoms_range[1]))
                        if invalid:
                            break
                    if invalid:
                        new_cg_pair=None
            if new_cg_pair is not None:
                new_pair_tps=[TrajectoryPoint(cg=new_cg) for new_cg in new_cg_pair]
                # If we use parallel tempering then we'll additionally try to shuffle the new molecules to make their
                # distribution among temperatures more physical.
                if self.beta is not None:
                    new_pair_tps, order_prob_balance=self.tp_pair_shuffle(new_pair_tps, old_ids)
                    prob_balance+=order_prob_balance
                    # Randomly assign new_pair_tps by old_ids (relevant for parallel tempering).
                    new_pair_tps, new_shuffle_prob_balance=self.tp_pair_shuffle(new_pair_tps, old_ids)
                    prob_balance+=new_shuffle_prob_balance-np.log(self.tp_pair_failed_shuffle_prob(old_cg_pair, old_ids))
                
                accepted=self.accept_reject_move(new_pair_tps, prob_balance, replica_ids=old_ids)
            if accepted:
                self.num_accepted_cross_couplings+=1
    def tp_pair_shuffle(self, tp_pair, replica_ids):
        prob_failed_shuffle=self.tp_pair_failed_shuffle_prob(tp_pair, replica_ids)
        do_shuffle=(random.random()>prob_failed_shuffle)
        output=tp_pair
        cur_prob=prob_failed_shuffle
        if do_shuffle:
            output=output[::-1]
            cur_prob=1.0-cur_prob
        return output, np.log(cur_prob)
    def tp_pair_failed_shuffle_prob(self, tp_pair, replica_ids):
        shuffle_pot_change=0.0
        if self.beta is not None:
            multiplier=1.0
            for replica_ids_shuffled in [replica_ids, replica_ids[::-1]]:
                shuffle_pot_change+=sum(self.beta[replica_id]*tp.calc_min_function(self.min_function) for replica_id, tp in zip(replica_ids_shuffled, tp_pair))*multiplier
                multiplier*=-1
        return (1.0+np.exp(-shuffle_pot_change))**(-1)
    def parallel_tempering(self, num_attempted_changes=1, change_histogram=True):
        for attempted_change_counter in range(num_attempted_changes):
            old_ids=np.random.randint(self.num_replicas, size=(2,))
            new_pair_tps=[self.cur_tps[old_id] for old_id in old_ids][::-1]
            self.accept_reject_move(new_pair_tps, 0.0, replica_ids=old_ids)
    # Genetic algorithms.
    def genetic_change_molecule_list(self, mol_list, num_attempted_changes=1, mol_format=None, randomized_change_params=None):
        output=copy.copy(mol_list)
        self.cur_tps=[TrajectoryPoint(egc=conv_func_mol_egc(mol, mol_format=mol_format)) for mol in mol_list]
        self.num_replicas=len(self.cur_tps)
        self.genetic_MC_step(num_attempted_changes=num_attempted_changes, randomized_change_params=randomized_change_params)
        return [conv_func_mol_egc(tp.egc, mol_format=mol_format, forward=False) for tp in self.cur_tps]
    # If we want to re-merge two different trajectories.
    def combine_with(self, other_rw):
        for tp in other_rw.explored_egcs:
            if tp in self.explored_egcs:
                tp_index=self.explored_egcs.index(tp)
                self.explored_egcs[tp_index].num_visits+=tp.num_visits
            else:
                self.explored_egcs.add(tp)
    # Procedures for biasing the potential.
    def explored_index_add(self, tp):
        if tp not in self.explored_egcs:
            self.explored_egcs.add(tp)
        return self.explored_egcs.index(tp)
    def biasing_potential(self, tp):
        if self.bias_coeff is None:
            return 0.0
        else:
            tp_index=self.explored_index_add(tp)
            return self.explored_egcs[tp_index].num_visits*self.bias_coeff
    def update_histogram(self):
        for replica_id in range(self.num_replicas):
            tp_index=self.explored_index_add(self.cur_tps[replica_id])
            self.explored_egcs[tp_index].num_visits+=1

def merge_random_walks(*rw_list):
    output=rw_list[0]
    for rw in rw_list[1:]:
        output.combine_trajectory(rw)
    return output
