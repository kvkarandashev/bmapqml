# TODO For forward and backward probabilities, comment more on where different signs come from.
from sortedcontainers import SortedList
from .ext_graph_compound import ExtGraphCompound
from .modify import atom_replacement_possibilities, atom_removal_possibilities, chain_addition_possibilities, bond_change_possibilities, valence_change_possibilities,\
                        add_heavy_atom_chain, remove_heavy_atom, replace_heavy_atom, change_bond_order, change_valence, randomized_cross_coupling
from .utils import rdkit_to_egc, egc_to_rdkit
from .valence_treatment import sorted_tuple, connection_forbidden
from .periodic import element_name
import random, copy
import numpy as np


# TODO 1. make default values for randomized change parameters work. 2. Add atoms with bond order more than one already?
# TODO 3. keep_histogram option needs more testing.

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
        return change_function(egc_in, *final_possibility_val[:2], possibility_label, resonance_structure_id=final_possibility_val[-1])
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
        if len(corr_prob_choice.keys())==0:
            raise Exception("Something is wrong: encountered a molecule that cannot be changed")
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

    if new_egc is None:
        return None, None

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
    def __init__(self, egc=None, cg=None, num_visits=None):
        if egc is None:
            if cg is not None:
                egc=ExtGraphCompound(chemgraph=cg)
        self.egc=egc

        if num_visits is not None:
            num_visits=copy.deecopy(num_visits)
        self.num_visits=num_visits
        self.visit_step_ids=None

        self.first_MC_step_encounter=None
        self.first_global_MC_step_encounter=None

        # Information for keeping detailed balance.
        self.bond_order_change_possibilities=None
        self.nuclear_charge_change_possibilities=None
        self.atom_removal_possibilities=None
        self.valence_change_possibilities=None
        self.chain_addition_possibilities=None

        self.calculated_data={}

    # TO-DO better way to write this?
    def init_possibility_info(self, bond_order_changes=[-1, 1], possible_elements=['C'], restricted_tps=None, **other_kwargs):
        # self.bond_order_change_possibilities is None - to check whether the init_* procedure has been called before.
        # self.egc.chemgraph.canonical_permutation - to check whether egc.chemgraph.changed() has been called.
        if (self.bond_order_change_possibilities is None) or (self.egc.chemgraph.canonical_permutation is None): # The latter check is made j
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
            if restricted_tps is not None:
                self.clear_possibility_info(restricted_tps)

    def clear_possibility_info(self, restricted_tps):
        for poss_func, poss_dict in self.possibilities().items():
            poss_key_id=0
            poss_keys=list(poss_dict.keys())
            while poss_key_id != len(poss_keys):
                poss_key=poss_keys[poss_key_id]
                poss_list=poss_dict[poss_key]

                poss_id=0
                while poss_id != len(poss_list):
                    result=egc_change_func(self.egc, poss_key, poss_list[poss_id], poss_func)
                    if (result is None) or (TrajectoryPoint(egc=result) in restricted_tps):
                        poss_id+=1
                    else:
                        del(poss_list[poss_id])
                if len(poss_list)==0:
                    del(poss_dict[poss_key])
                    del(poss_keys[poss_key_id])
                else:
                    poss_key_id+=1


    def possibilities(self):
        return {add_heavy_atom_chain : self.chain_addition_possibilities,
                    replace_heavy_atom : self.nuclear_charge_change_possibilities,
                    remove_heavy_atom : self.atom_removal_possibilities,
                    change_bond_order : self.bond_order_change_possibilities,
                    change_valence : self.valence_change_possibilities}

    def calc_or_lookup(self, func_dict, args_dict=None, kwargs_dict=None):
        output={}
        for quant_name in func_dict.keys():
            if quant_name not in self.calculated_data:
                if args_dict is None:
                    args=()
                else:
                    args=args_dict[quant_name]
                if kwargs_dict is None:
                    kwargs={}
                else:
                    kwargs=kwargs_dict[quant_name]
                self.calculated_data[quant_name]=func_dict[quant_name](self, *args, **kwargs)
            output[quant_name]=self.calculated_data[quant_name]
        return output

    def visit_num(self, replica_id):
        if self.num_visits is None:
            return 0
        else:
            return self.num_visits[replica_id]

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

# Auxiliary class for more convenient maintenance of candidate compound list.
class CandidateCompound:
    def __init__(self, tp, func_val):
        self.tp=tp
        self.func_val=func_val
    def __eq__(self, cc2):
        return (self.tp == cc2.tp)
    def __gt__(self, cc2):
        if self==cc2:
            return False
        else:
            return self.func_val > cc2.func_val
    def __lt__(self, cc2):
        if self==cc2:
            return False
        else:
            return self.func_val < cc2.func_val
    def __str__(self):
        return "(CandidateCompound,func_val:"+str(self.func_val)+",ChemGraph:"+str(self.tp.egc.chemgraph)
    def __repr__(self):
        return str(self)

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

def random_pair(arr_length):
    return random.sample(range(arr_length), 2)

# Class that generates a trajectory over chemical space.
class RandomWalk:
    def __init__(self, init_egcs=None, bias_coeff=None, vbeta_bias_coeff=None, randomized_change_params={}, starting_histogram=None, conserve_stochiometry=False,
                    bound_enforcing_coeff=1.0, keep_histogram=False, betas=None, min_function=None, num_replicas=None, no_exploration=False,
                    no_exploration_smove_adjust=False, restricted_tps=None, min_function_name="MIN_FUNCTION", num_saved_candidates=None,
                    keep_full_trajectory=False):
        """
        betas : values of beta used in the extended tempering ensemble; "None" corresponds to a virtual beta (greedily minimized replica).
        bias_coeff : biasing potential applied to push real beta replicas out of local minima
        vbeta_bias_coeff : biasing potential applied to push virtual beta replicas out of local minima
        """
        self.num_replicas=num_replicas
        self.betas=betas
        if self.num_replicas is None:
            if self.betas is not None:
                self.num_replicas=len(self.betas)
            else:
                if isinstance(init_egcs, list):
                    self.num_replicas=len(init_egcs)


        if init_egcs is None:
            self.cur_tps=None
        else:
            self.cur_tps=[TrajectoryPoint(egc=init_egc) for init_egc in init_egcs]

        self.keep_full_trajectory=keep_full_trajectory

        self.MC_step_counter=0
        self.global_MC_step_counter=0

        if isinstance(self.betas, list):
            assert(len(self.betas)==self.num_replicas)

        if starting_histogram is None:
            if keep_histogram:
                self.histogram=SortedList()
                if self.cur_tps is not None:
                    self.update_histogram(list(range(self.num_replicas)))
            else:
                self.histogram=None
        else:
            self.histogram=starting_histogram

        self.no_exploration=no_exploration
        if self.no_exploration:
            if restricted_tps is None:
                raise Exception
            else:
                self.restricted_tps=restricted_tps
                self.no_exploration_smove_adjust=no_exploration_smove_adjust

        self.bias_coeff=bias_coeff
        self.vbeta_bias_coeff=vbeta_bias_coeff

        self.randomized_change_params=randomized_change_params
        self.init_randomized_change_params(randomized_change_params)

        self.keep_histogram=keep_histogram

        self.bound_enforcing_coeff=bound_enforcing_coeff

        # TODO previous implementation deprecated; perhaps re-implement as a ``soft'' constraint?
        # Or make a special dictionnary of changes that conserve stochiometry.
        self.conserve_stochiometry=conserve_stochiometry

        self.hydrogen_nums=None

        self.min_function=min_function
        if self.min_function is not None:
            self.min_function_name=min_function_name
            self.min_function_dict={self.min_function_name : self.min_function}
        self.num_saved_candidates=num_saved_candidates
        if self.num_saved_candidates is not None:
            self.saved_candidates=[]

        # For storing statistics on move success.
        self.num_attempted_cross_couplings=0
        self.num_accepted_cross_couplings=0
        self.moves_since_changed=np.zeros((self.num_replicas,), dtype=int)

    def init_randomized_change_params(self, randomized_change_params=None):
        if randomized_change_params is not None:
            self.randomized_change_params=randomized_change_params
            if "forbidden_bonds" in self.randomized_change_params:
                self.randomized_change_params["forbidden_bonds"]=tidy_forbidden_bonds(self.randomized_change_params["forbidden_bonds"])
        self.used_randomized_change_params=copy.deepcopy(self.randomized_change_params)

        if self.no_exploration:
            if self.no_exploration_smove_adjust:
                self.used_randomized_change_params["restricted_tps"]=self.restricted_tps

    # Acceptance rejection rules.
    def accept_reject_move(self, new_tps, prob_balance, replica_ids=[0]):
        self.MC_step_counter+=1

        accepted=self.acceptance_rule(new_tps, prob_balance, replica_ids=replica_ids)
        if accepted:
            for new_tp, replica_id in zip(new_tps, replica_ids):
                self.cur_tps[replica_id]=new_tp
                self.moves_since_changed[replica_id]=0
        else:
            for replica_id in replica_ids:
                self.moves_since_changed[replica_id]+=1


        if self.keep_histogram:
            self.update_histogram(replica_ids)
        if self.num_saved_candidates is not None:
            self.update_saved_candidates()

        return accepted

    def acceptance_rule(self, new_tps, prob_balance, replica_ids=[0]):

        if self.no_exploration:
            for new_tp in new_tps:
                if new_tp not in self.restricted_tps:
                    return False
        if self.histogram is not None:
            for new_tp_id in range(len(new_tps)):
                histogram_new_tp_id=self.histogram_index_add(new_tps[new_tp_id])
                new_tps[new_tp_id]=self.histogram[histogram_new_tp_id]

        delta_pot=prob_balance

        prev_tps=[self.cur_tps[replica_id] for replica_id in replica_ids]

        if self.min_function is not None:
            if self.virtual_beta_present(replica_ids):
                delta_pot=.0
                trial_vals=[]
                old_vals=[]
                for replica_id, new_tp, prev_tp in zip(replica_ids, new_tps, prev_tps):
                    if self.virtual_beta_id(replica_id):
                        trial_val=self.eval_min_func(new_tp)
                        old_val=self.eval_min_func(prev_tp)

                        if self.vbeta_bias_coeff is not None:
                            trial_val+=self.vbeta_bias_coeff*new_tp.visit_num(replica_id)
                            old_val+=self.vbeta_bias_coeff*prev_tp.visit_num(replica_id)

                        trial_vals.append(trial_val)
                        old_vals.append(old_val)
                return (min(trial_vals)<min(old_vals))
            else:
                for replica_id, new_tp, prev_tp in zip(replica_ids, new_tps, prev_tps):
                    delta_pot+=self.betas[replica_id]*(self.eval_min_func(new_tp)-self.eval_min_func(prev_tp))

        for replica_id, new_tp, prev_tp in zip(replica_ids, new_tps, prev_tps):
            if self.bias_coeff is not None:
                delta_pot+=self.biasing_potential(new_tp, replica_id)-self.biasing_potential(prev_tp, replica_id)
            if self.bound_enforcing_coeff is not None:
                delta_pot+=self.bound_enforcing_pot(new_tp, replica_id)-self.bound_enforcing_pot(prev_tp, replica_id)

        if delta_pot<.0:
            return True
        else:
            return (random.random()<np.exp(-delta_pot))

    def virtual_beta_present(self, beta_ids):
        return any([self.virtual_beta_id(beta_id) for beta_id in beta_ids])

    def virtual_beta_id(self, beta_id):
        return (self.betas[beta_id] is None)

    def bound_enforcing_pot(self, tp, replica_id):
        return self.bound_enforcing_coeff*self.bound_outlie(tp.egc, replica_id)

    def bound_outlie(self, egc, replica_id):
        output=0
        if self.hydrogen_nums is not None:
            if egc.chemgraph.tot_nhydrogens() != self.hydrogen_nums[replica_id]:
                output+=abs(egc.chemgraph.tot_nhydrogens()-self.hydrogen_nums[replica_id])
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

    # TODO do we still need Metropolis_rejection_prob? Does not appear anywhere.
    def tp_pair_order_prob(self, replica_ids, tp_pair=None, Metropolis_rejection_prob=False):
        if tp_pair is None:
            tp_pair=[self.cur_tps[replica_id] for replica_id in replica_ids]
        mfunc_vals=[self.eval_min_func(tp) for tp in tp_pair]
        if self.virtual_beta_present(replica_ids):
            if all([(self.betas[replica_id] is None) for replica_id in replica_ids]):
                return .5
            else:
                min_arg=np.argmin(mfunc_vals)
                if self.betas[replica_ids[min_arg]] is None:
                    return 1.
                else:
                    return .0
        else:
            delta_pot=-(self.betas[replica_ids[0]]-self.betas[replica_ids[1]])*(self.eval_min_func(tp_pair[0])-self.eval_min_func(tp_pair[1]))
            if Metropolis_rejection_prob:
                if delta_pot<.0:
                    return 0.
                else:
                    return 1.-np.exp(-delta_pot)
            else:
                return (1.0+np.exp(-delta_pot))**(-1)

    # Basic move procedures.
    def MC_step(self, replica_id=0, **dummy_kwargs):
        changed_tp=self.cur_tps[replica_id]
        changed_tp.init_possibility_info(**self.used_randomized_change_params)
        new_tp, prob_balance=randomized_change(changed_tp, **self.used_randomized_change_params)
        if new_tp is None:
            return False
        accepted=self.accept_reject_move([new_tp], prob_balance, replica_ids=[replica_id])
        return accepted

    def genetic_MC_step(self, replica_ids):
        old_cg_pair=[self.cur_tps[replica_id].egc.chemgraph for replica_id in replica_ids]
        new_cg_pair, prob_balance=randomized_cross_coupling(old_cg_pair, **self.used_randomized_change_params)
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
        if new_cg_pair is None:
            return False
        new_pair_tps=[TrajectoryPoint(cg=new_cg) for new_cg in new_cg_pair]
        if self.min_function is not None:
            new_pair_shuffle_prob=self.tp_pair_order_prob(replica_ids, tp_pair=new_pair_tps)
            if random.random()>new_pair_shuffle_prob: # shuffle
                new_pair_shuffle_prob=1.-new_pair_shuffle_prob
                new_pair_tps=new_pair_tps[::-1]
            if self.virtual_beta_present(replica_ids):
                old_pair_shuffle_prob=1. # will be ignored anyway
            else:
                old_pair_shuffle_prob=self.tp_pair_order_prob(replica_ids) 
            prob_balance+=np.log(new_pair_shuffle_prob/old_pair_shuffle_prob)
        accepted=self.accept_reject_move(new_pair_tps, prob_balance, replica_ids=replica_ids)
        if accepted:
            self.num_accepted_cross_couplings+=1
        return accepted

    # Procedures for changing entire list of Trajectory Points at once.
    def MC_step_all(self, **mc_step_kwargs):
        output=[]
        for replica_id in range(self.num_replicas):
            output.append(self.MC_step(**mc_step_kwargs, replica_id=replica_id))
        return output

    def genetic_MC_step_all(self, num_genetic_tries=1, randomized_change_params=None, **dummy_kwargs):
        self.init_randomized_change_params(randomized_change_params=randomized_change_params)
        for attempted_change_counter in range(num_genetic_tries):
            changed_replica_ids=random_pair(self.num_replicas)
            self.genetic_MC_step(changed_replica_ids)

    def parallel_tempering(self, num_parallel_tempering_tries=1, **dummy_kwargs):

        if self.min_function is not None:
            for attempted_change_counter in range(num_parallel_tempering_tries):
                old_ids=random_pair(self.num_replicas)
                trial_tps=[self.cur_tps[old_ids[1]], self.cur_tps[old_ids[0]]]
                accepted=self.accept_reject_move(trial_tps, .0, replica_ids=old_ids)

    def global_random_change(self, prob_dict={"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}, **other_kwargs):
        self.global_MC_step_counter+=1
        cur_procedure=random.choices(list(prob_dict), weights=list(prob_dict.values()))[0]
        if cur_procedure=="simple":
            self.MC_step_all(**other_kwargs)
            return
        if cur_procedure=="genetic":
            self.genetic_MC_step_all(**other_kwargs)
            return
        if cur_procedure=="tempering":
            self.parallel_tempering(**other_kwargs)
            return
        raise Exception("Unknown option picked.")

    # For convenient interfacing with other scripts.
    def convert_to_current_egcs(self, mol_in_list, mol_format=None):
        self.cur_tps=[TrajectoryPoint(egc=conv_func_mol_egc(mol_in, mol_format=mol_format)) for mol_in in mol_in_list]
        self.num_replicas=len(self.cur_tps)
        if self.betas is not None:
            assert(len(self.betas)==self.num_replicas)

    def converted_current_egcs(self, mol_format=None):
        return [conv_func_mol_egc(cur_tp.egc, mol_format=mol_format, forward=False) for cur_tp in self.cur_tps]

    def change_molecule(self, mol_in, **change_molecule_params):
        return self.change_molecule_list([mol_in], **change_molecule_params)[0]

    def change_molecule_list(self, mol_in_list, randomized_change_params=None, mol_format=None):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.init_randomized_change_params(randomized_change_params=None)
        moves_accepted=self.MC_step_all()
        return self.converted_current_egcs(mol_format=mol_format)

    def parallel_tempering_molecule_list(self, mol_in_list, mol_format=None, **other_kwargs):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.parallel_tempering(**other_kwargs)
        return self.converted_current_egcs(mol_format=mol_format)

    def genetic_change_molecule_list(self, mol_in_list, mol_format=None, **other_kwargs):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.genetic_MC_step_all(**other_kwargs)
        return self.converted_current_egcs(mol_format=mol_format)

    # Either evaluate minimized function or look it up.
    def eval_min_func(self, tp):
        return tp.calc_or_lookup(self.min_function_dict)[self.min_function_name]

    # If we want to re-merge two different trajectories.
    def combine_with(self, other_rw):
        for tp in other_rw.histogram:
            if tp in self.histogram:
                tp_index=self.histogram.index(tp)
                self.histogram[tp_index].num_visits+=tp.num_visits
            else:
                self.histogram.add(tp)
    # Procedures for biasing the potential.
    def histogram_index_add(self, tp):
        if tp not in self.histogram:
            self.histogram.add(tp)
        return self.histogram.index(tp)

    def biasing_potential(self, tp, replica_id):
        if self.bias_coeff is None:
            return 0.0
        else:
            tp_index=self.histogram_index_add(tp)
            return self.histogram[tp_index].visit_num(replica_id)*self.bias_coeff

    def update_histogram(self, replica_ids):
        for replica_id in replica_ids:
            tp_index=self.histogram_index_add(self.cur_tps[replica_id])
            if self.histogram[tp_index].num_visits is None:
                self.histogram[tp_index].num_visits=np.zeros((self.num_replicas,), dtype=int)
                self.histogram[tp_index].first_MC_step_encounter=self.MC_step_counter
                self.histogram[tp_index].first_global_MC_step_encounter=self.global_MC_step_counter

            self.histogram[tp_index].num_visits[replica_id]+=1

            if self.keep_full_trajectory:
                if self.histogram[tp_index].visit_step_ids is None:
                    self.histogram[tp_index].visit_step_ids=[[] for replica_id in range(self.num_replicas)]
                self.histogram[tp_index].visit_step_ids[replica_id].append(self.global_MC_step_counter)

    def clear_histogram_visit_data(self):
        for tp in self.histogram:
            tp.num_visits=None
            if self.keep_full_trajectory:
                tp.visit_step_ids=None

    def update_saved_candidates(self):
        for tp in self.cur_tps:
            if self.min_function_name in tp.calculated_data:
                new_cc=CandidateCompound(tp, tp.calculated_data[self.min_function_name])
                if new_cc not in self.saved_candidates:
                    self.saved_candidates.append(new_cc)
        self.saved_candidates.sort()
        if len(self.saved_candidates)>self.num_saved_candidates:
            del(self.saved_candidates[self.num_saved_candidates:])

    # Some properties for more convenient trajectory analysis.
    def ordered_trajectory(self):
        output=[]
        for tp_ids in self.ordered_trajectory_ids():
            output.append([self.histogram[tp_id] for tp_id in tp_ids])
        return output
    def ordered_trajectory_ids(self):
        assert(self.keep_full_trajectory)
        output=np.zeros((self.global_MC_step_counter, self.num_replicas), dtype=int)
        for tp_id, tp in enumerate(self.histogram):
            if tp.visit_step_ids is not None:
                for replica_id, replica_visits in enumerate(tp.visit_step_ids):
                    for replica_visit in replica_visits:
                        output[replica_visit-1, replica_id]=tp_id
        return output


def merge_random_walks(*rw_list):
    output=rw_list[0]
    for rw in rw_list[1:]:
        output.combine_trajectory(rw)
    return output
