# TODO For forward and backward probabilities, comment more on where different signs come from.
# TODO 1. Account for statistical noise of input data. 2. Many theory levels?

from sortedcontainers import SortedList
from .ext_graph_compound import ExtGraphCompound
from .modify import (
    atom_replacement_possibilities,
    atom_removal_possibilities,
    chain_addition_possibilities,
    bond_change_possibilities,
    valence_change_possibilities,
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
    randomized_cross_coupling,
    egc_valid_wrt_change_params,
)
from .utils import rdkit_to_egc, egc_to_rdkit
from ..utils import dump2pkl, loadpkl, dump2tar, loadtar
from .valence_treatment import sorted_tuple, connection_forbidden, ChemGraph
from .periodic import element_name
import random, os
from copy import deepcopy
import numpy as np

np.seterr(all="raise")


def exp_wexceptions(val):
    try:
        return np.exp(val)
    except FloatingPointError:
        if val > 0.0:
            return np.inf
        else:
            return 0.0


# TODO 1. make default values for randomized change parameters work. 2. Add atoms with bond order more than one already?
# TODO 3. keep_histogram option needs more testing.

default_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
]

stochiometry_conserving_change_list = [change_bond_order, change_valence]

inverse_procedure = {
    add_heavy_atom_chain: remove_heavy_atom,
    remove_heavy_atom: add_heavy_atom_chain,
    replace_heavy_atom: replace_heavy_atom,
    change_bond_order: change_bond_order,
    change_valence: change_valence,
}

change_possibility_label = {
    add_heavy_atom_chain: "possible_elements",
    remove_heavy_atom: "possible_elements",
    replace_heavy_atom: "possible_elements",
    change_bond_order: "bond_order_changes",
    change_valence: None,
}


def inverse_possibility_label(change_function, possibility_label):
    if change_function is change_bond_order:
        return -possibility_label
    return possibility_label


def egc_change_func(egc_in, possibility_label, final_possibility_val, change_function):
    if change_function is change_bond_order:
        return change_function(
            egc_in,
            *final_possibility_val[:2],
            possibility_label,
            resonance_structure_id=final_possibility_val[-1]
        )
    if change_function is remove_heavy_atom:
        return change_function(egc_in, final_possibility_val)
    if change_function is change_valence:
        return change_function(egc_in, possibility_label, final_possibility_val)
    true_possibility_label = possibility_label
    if change_function is add_heavy_atom_chain:
        true_possibility_label = [possibility_label]
    return change_function(egc_in, final_possibility_val, true_possibility_label)


def available_options_prob_norm(dict_in):
    output = 0.0
    for i in list(dict_in):
        if len(dict_in[i]) != 0:
            output += 1.0
    return output


def random_choice_from_dict(possibilities, choices=None, get_probability_of=None):
    prob_sum = 0.0
    corr_prob_choice = {}
    if choices is None:
        choices = list(possibilities.keys())
    for choice in list(choices):
        if (choice not in list(possibilities)) or (len(possibilities[choice]) == 0):
            continue
        if isinstance(choices, dict):
            prob = choices[choice]
        else:
            prob = 1.0
        prob_sum += prob
        corr_prob_choice[choice] = prob
    if get_probability_of is None:
        if len(corr_prob_choice.keys()) == 0:
            raise Exception(
                "Something is wrong: encountered a molecule that cannot be changed"
            )
        final_choice = random.choices(
            list(corr_prob_choice.keys()), list(corr_prob_choice.values())
        )[0]
        final_prob_log = np.log(corr_prob_choice[final_choice] / prob_sum)
        return final_choice, possibilities[final_choice], final_prob_log
    else:
        return possibilities[get_probability_of], np.log(
            corr_prob_choice[get_probability_of] / prob_sum
        )


def lookup_or_none(dict_in, key):
    if key in dict_in:
        return dict_in[key]
    else:
        return None


def tp_or_chemgraph(tp):
    if isinstance(tp, ChemGraph):
        return tp
    else:
        return tp.chemgraph()


class TrajectoryPoint:
    def __init__(
        self,
        egc: ExtGraphCompound or None = None,
        cg: ChemGraph or None = None,
        num_visits: int or None = None,
    ):
        """
        This class stores an ExtGraphCompound object along with all the infromation needed to preserve detailed balance of the random walk.
        egc : ExtGraphCompound object to be stored.
        cg : ChemGraph object used to define egc if the latter is None
        num_visits : initial numbers of visits to the trajectory
        """
        if egc is None:
            if cg is not None:
                egc = ExtGraphCompound(chemgraph=cg)
        self.egc = egc

        if num_visits is not None:
            num_visits = deepcopy(num_visits)
        self.num_visits = num_visits
        self.visit_step_ids = None
        self.visit_global_step_ids = None

        self.first_MC_step_encounter = None
        self.first_global_MC_step_encounter = None

        # Information for keeping detailed balance.
        self.bond_order_change_possibilities = None
        self.nuclear_charge_change_possibilities = None
        self.atom_removal_possibilities = None
        self.valence_change_possibilities = None
        self.chain_addition_possibilities = None

        self.calculated_data = {}

    # TO-DO better way to write this?
    def init_possibility_info(
        self,
        bond_order_changes=[-1, 1],
        possible_elements=["C"],
        restricted_tps=None,
        **other_kwargs
    ):
        # self.bond_order_change_possibilities is None - to check whether the init_* procedure has been called before.
        # self.egc.chemgraph.canonical_permutation - to check whether egc.chemgraph.changed() has been called.
        if (self.bond_order_change_possibilities is None) or (
            self.egc.chemgraph.canonical_permutation is None
        ):  # The latter check is made j
            self.bond_order_change_possibilities = {}
            for bond_order_change in bond_order_changes:
                cur_possibilities = bond_change_possibilities(
                    self.egc, bond_order_change, **other_kwargs
                )
                if len(cur_possibilities) != 0:
                    self.bond_order_change_possibilities[
                        bond_order_change
                    ] = cur_possibilities
            self.nuclear_charge_change_possibilities = {}
            self.atom_removal_possibilities = {}
            self.chain_addition_possibilities = {}
            for possible_element in possible_elements:
                repl_possibilities = atom_replacement_possibilities(
                    self.egc, possible_element, **other_kwargs
                )

                if len(repl_possibilities) != 0:
                    self.nuclear_charge_change_possibilities[
                        possible_element
                    ] = repl_possibilities
                removal_possibilities = atom_removal_possibilities(
                    self.egc,
                    deleted_atom=possible_element,
                    only_end_atoms=True,
                    **other_kwargs
                )
                if len(removal_possibilities) != 0:
                    self.atom_removal_possibilities[
                        possible_element
                    ] = removal_possibilities
                addition_possibilities = chain_addition_possibilities(
                    self.egc, chain_starting_element=possible_element, **other_kwargs
                )
                if len(addition_possibilities) != 0:
                    self.chain_addition_possibilities[
                        possible_element
                    ] = addition_possibilities
            self.valence_change_possibilities = valence_change_possibilities(self.egc)
            if restricted_tps is not None:
                self.clear_possibility_info(restricted_tps)

    def clear_possibility_info(self, restricted_tps):
        for poss_func, poss_dict in self.possibilities().items():
            poss_key_id = 0
            poss_keys = list(poss_dict.keys())
            while poss_key_id != len(poss_keys):
                poss_key = poss_keys[poss_key_id]
                poss_list = poss_dict[poss_key]

                poss_id = 0
                while poss_id != len(poss_list):
                    result = egc_change_func(
                        self.egc, poss_key, poss_list[poss_id], poss_func
                    )
                    if (result is None) or (
                        TrajectoryPoint(egc=result) in restricted_tps
                    ):
                        poss_id += 1
                    else:
                        del poss_list[poss_id]
                if len(poss_list) == 0:
                    del poss_dict[poss_key]
                    del poss_keys[poss_key_id]
                else:
                    poss_key_id += 1

    def possibilities(self):
        return {
            add_heavy_atom_chain: self.chain_addition_possibilities,
            replace_heavy_atom: self.nuclear_charge_change_possibilities,
            remove_heavy_atom: self.atom_removal_possibilities,
            change_bond_order: self.bond_order_change_possibilities,
            change_valence: self.valence_change_possibilities,
        }

    def clear_possibility_info(self):
        self.chain_addition_possibilities = None
        self.nuclear_charge_change_possibilities = None
        self.atom_removal_possibilities = None
        self.bond_order_change_possibilities = None
        self.valence_change_possibilities = None

    def calc_or_lookup(self, func_dict, args_dict=None, kwargs_dict=None):
        output = {}
        for quant_name in func_dict.keys():
            if quant_name not in self.calculated_data:
                if args_dict is None:
                    args = ()
                else:
                    args = args_dict[quant_name]
                if kwargs_dict is None:
                    kwargs = {}
                else:
                    kwargs = kwargs_dict[quant_name]
                self.calculated_data[quant_name] = func_dict[quant_name](
                    self, *args, **kwargs
                )
            output[quant_name] = self.calculated_data[quant_name]
        return output

    def visit_num(self, replica_id):
        if self.num_visits is None:
            return 0
        else:
            return self.num_visits[replica_id]

    def copy_extra_data_to(self, other_tp, linear_storage=False):
        """
        Copy all calculated data from self to other_tp.
        """
        for quant_name in self.calculated_data:
            if quant_name not in other_tp.calculated_data:
                other_tp.calculated_data[quant_name] = self.calculated_data[quant_name]
        # Dealing with making sure the order is preserved is too complicated.
        # if self.bond_order_change_possibilities is not None:
        #    if other_tp.bond_order_change_possibilities is None:
        #        other_tp.bond_order_change_possibilities = deepcopy(
        #            self.bond_order_change_possibilities
        #        )
        #        other_tp.chain_addition_possibilities = deepcopy(
        #            self.chain_addition_possibilities
        #        )
        #        other_tp.nuclear_charge_change_possibilities = deepcopy(
        #            self.nuclear_charge_change_possibilities
        #        )
        #        other_tp.atom_removal_possibilities = deepcopy(
        #            self.atom_removal_possibilities
        #        )
        #        other_tp.valence_change_possibilities = deepcopy(
        #            self.valence_change_possibilities
        #        )
        self.egc.chemgraph.copy_extra_data_to(
            other_tp.egc.chemgraph, linear_storage=linear_storage
        )

    def chemgraph(self):
        return self.egc.chemgraph

    def __lt__(self, tp2):
        return self.chemgraph() < tp_or_chemgraph(tp2)

    def __gt__(self, tp2):
        return self.chemgraph() > tp_or_chemgraph(tp2)

    def __eq__(self, tp2):
        return self.chemgraph() == tp_or_chemgraph(tp2)

    def __str__(self):
        return str(self.egc)

    def __repr__(self):
        return str(self)


# Auxiliary class for more convenient maintenance of candidate compound list.
class CandidateCompound:
    def __init__(self, tp, func_val):
        self.tp = tp
        self.func_val = func_val

    def __eq__(self, cc2):
        return self.tp == cc2.tp

    def __gt__(self, cc2):
        if self == cc2:
            return False
        else:
            return self.func_val > cc2.func_val

    def __lt__(self, cc2):
        if self == cc2:
            return False
        else:
            return self.func_val < cc2.func_val

    def __str__(self):
        return (
            "(CandidateCompound,func_val:"
            + str(self.func_val)
            + ",ChemGraph:"
            + str(self.tp.egc.chemgraph)
        )

    def __repr__(self):
        return str(self)


def tidy_forbidden_bonds(forbidden_bonds):
    if forbidden_bonds is None:
        return None
    output = SortedList()
    for fb in forbidden_bonds:
        output.add(sorted_tuple(*fb))
    return output


def has_forbidden_bonds(egc, forbidden_bonds=None):
    adjmat = egc.true_adjmat()
    for i1, nc1 in enumerate(egc.nuclear_charges[: egc.num_heavy_atoms()]):
        for i2, nc2 in enumerate(egc.nuclear_charges[:i1]):
            if adjmat[i1, i2] != 0:
                if connection_forbidden(nc1, nc2, forbidden_bonds):
                    return True


mol_egc_converter = {"rdkit": (rdkit_to_egc, egc_to_rdkit)}


def conv_func_mol_egc(inp, forward=True, mol_format=None):
    if mol_format is None:
        return inp
    else:
        if forward:
            conv_func_id = 0
        else:
            conv_func_id = 1
        return mol_egc_converter[mol_format][conv_func_id](inp)


#   Auxiliary exception classes.
class SoftExitCalled(Exception):
    """
    Exception raised by RandomWalk object if soft exit request is passed.
    """

    pass


class InvalidStartingMolecules(Exception):
    """
    Exception raised by RandomWalk object if it is initialized with starting molecules that do not fit the parameters for random change.
    """

    pass


class DataUnavailable(Exception):
    """
    Raised if data not available in a histogram is referred to.
    """

    pass


from types import FunctionType


def randomized_change(
    tp: TrajectoryPoint,
    change_prob_dict=default_change_list,
    visited_tp_list: list or None = None,
    **other_kwargs
):
    """
    Randomly modify a TrajectoryPoint object.
    visited_tp_list : list of TrajectoryPoint objects for which data is available.
    """
    cur_change_procedure, possibilities, total_forward_prob = random_choice_from_dict(
        tp.possibilities(), change_prob_dict
    )
    possibility_dict_label = change_possibility_label[cur_change_procedure]
    possibility_dict = lookup_or_none(other_kwargs, possibility_dict_label)
    possibility_label, change_possibilities, forward_prob = random_choice_from_dict(
        possibilities, possibility_dict
    )
    total_forward_prob += forward_prob - np.log(len(change_possibilities))
    final_possibility_val = random.choice(change_possibilities)

    new_egc = egc_change_func(
        tp.egc, possibility_label, final_possibility_val, cur_change_procedure
    )

    if new_egc is None:
        return None, None

    new_tp = TrajectoryPoint(egc=new_egc)
    if visited_tp_list is not None:
        if new_tp in visited_tp_list:
            tp_id = visited_tp_list.index(new_tp)
            visited_tp_list[tp_id].copy_extra_data_to(new_tp)

    new_tp.init_possibility_info(**other_kwargs)

    # Calculate the chances of doing the inverse operation
    inv_proc = inverse_procedure[cur_change_procedure]
    inverse_possibilities, total_inverse_prob = random_choice_from_dict(
        new_tp.possibilities(), change_prob_dict, get_probability_of=inv_proc
    )
    if cur_change_procedure is change_valence:
        inverse_prob = -np.log(len(new_tp.possibilities()[cur_change_procedure]))
        inverse_final_possibilities = change_possibilities
    else:
        inv_poss_dict_label = change_possibility_label[inv_proc]
        inv_poss_dict = lookup_or_none(other_kwargs, inv_poss_dict_label)
        if cur_change_procedure is replace_heavy_atom:
            inverse_pl = element_name[
                tp.egc.chemgraph.hatoms[final_possibility_val].ncharge
            ]
        else:
            inverse_pl = inverse_possibility_label(
                cur_change_procedure, possibility_label
            )
        inverse_final_possibilities, inverse_prob = random_choice_from_dict(
            inverse_possibilities, inv_poss_dict, get_probability_of=inverse_pl
        )
    total_inverse_prob += inverse_prob - np.log(len(inverse_final_possibilities))

    return new_tp, total_forward_prob - total_inverse_prob


class RandomWalk:
    def __init__(
        self,
        init_egcs: list or None = None,
        bias_coeff: float or None = None,
        vbeta_bias_coeff: float or None = None,
        bias_pot_all_replicas: bool = True,
        randomized_change_params: dict = {},
        starting_histogram: list or None = None,
        conserve_stochiometry: bool = False,
        bound_enforcing_coeff: float or None = None,
        keep_histogram: bool = False,
        histogram_save_rejected: bool = True,
        betas: list or None = None,
        min_function: FunctionType or None = None,
        num_replicas: int or None = None,
        no_exploration: bool = False,
        no_exploration_smove_adjust: bool = False,
        restricted_tps: list or None = None,
        min_function_name: str = "MIN_FUNCTION",
        num_saved_candidates: int or None = None,
        keep_full_trajectory: bool = False,
        restart_file: str or None = None,
        make_restart_frequency: int or None = None,
        soft_exit_check_frequency: int or None = None,
        delete_temp_data: list or None = None,
        max_histogram_size: int or None = None,
        histogram_dump_file_prefix: str = "",
        track_histogram_size: bool = False,
        visit_num_count_acceptance: bool = False,
        linear_storage: bool = False,
        compress_restart: bool = False,
    ):
        """
        Class that generates a trajectory over chemical space.
        init_egcs : initial positions of the simulation, in ExtGraphCompound format.
        betas : values of beta used in the extended tempering ensemble; "None" corresponds to a virtual beta (greedily minimized replica).
        bias_coeff : biasing potential applied to push real beta replicas out of local minima
        vbeta_bias_coeff : biasing potential applied to push virtual beta replicas out of local minima
        bias_pot_all_replicas : whether the biasing potential is calculated from sum of visits of all replicas rather than the replica considered
        min_function : minimized function
        min_function_name : name of minimized function (the label used for minimized function value in TrajectoryPoint object's calculated_data)
        keep_histogram : store information about all considered molecules; mandatory for using biasing potentials
        histogram_save_rejected : if True then both accepted and rejected chemical graphs are saved into the histogram
        num_saved_candidates : if not None determines how many best candidates are kept in the saved_candidates attributes
        keep_full_trajectory : save not just number of times a trajectory point was visited, but also all steps ids when the step was made
        restart_file : name of restart file to which the object is dumped at make_restart
        make_restart_frequency : if not None object will call make_restart each make_restart_frequency global steps.
        soft_exit_check_frequency : if not None the code will check presence of "EXIT" in the running directory each soft_exit_check_frequency steps; if "EXIT" exists the object calls make_restart and raises SoftExitCalled
        delete_temp_data : if not None after each minimized function evaluation for a TrajectoryPoint object self will delete TrajectoryPoint's calculated_data fields with those identifiers.
        max_histogram_size : if not None sets the maximal size for the histogram that, when exceeded, triggers dumping the histogram
        histogram_dump_file_prefix : sets the prefix from which the name of the pickle file where histogram is dumped if its maximal size is exceeded
        track_histogram_size : print current size of the histogram after each global MC step
        visit_num_count_acceptance : if True number of visit numbers (used in biasing potential) is counted during each accept_reject_move call rather than each global step
        linear_storage : whether objects saved to the histogram contain data whose size scales more than linearly with molecule size
        compress_restart : whether restart files are compressed by default
        """
        self.num_replicas = num_replicas
        self.betas = betas
        if self.num_replicas is None:
            if self.betas is not None:
                self.num_replicas = len(self.betas)
            else:
                if isinstance(init_egcs, list):
                    self.num_replicas = len(init_egcs)
                else:
                    self.num_replicas = 1

        self.keep_full_trajectory = keep_full_trajectory

        self.MC_step_counter = 0
        self.global_MC_step_counter = 0

        if isinstance(self.betas, list):
            assert len(self.betas) == self.num_replicas

        self.keep_histogram = keep_histogram
        self.histogram_save_rejected = histogram_save_rejected
        self.visit_num_count_acceptance = visit_num_count_acceptance
        self.linear_storage = linear_storage

        self.no_exploration = no_exploration
        if self.no_exploration:
            if restricted_tps is None:
                raise Exception
            else:
                self.restricted_tps = restricted_tps
                self.no_exploration_smove_adjust = no_exploration_smove_adjust

        self.bias_coeff = bias_coeff
        self.vbeta_bias_coeff = vbeta_bias_coeff
        self.bias_pot_all_replicas = bias_pot_all_replicas

        self.randomized_change_params = randomized_change_params
        self.init_randomized_change_params(randomized_change_params)

        self.bound_enforcing_coeff = bound_enforcing_coeff

        # TODO previous implementation deprecated; perhaps re-implement as a ``soft'' constraint?
        # Or make a special dictionnary of changes that conserve stochiometry.
        self.conserve_stochiometry = conserve_stochiometry

        self.hydrogen_nums = None

        self.min_function = min_function
        if self.min_function is not None:
            self.min_function_name = min_function_name
            self.min_function_dict = {self.min_function_name: self.min_function}
        self.num_saved_candidates = num_saved_candidates
        if self.num_saved_candidates is not None:
            self.saved_candidates = []

        # For storing statistics on move success.
        self.num_attempted_cross_couplings = 0
        self.num_accepted_cross_couplings = 0
        self.moves_since_changed = np.zeros((self.num_replicas,), dtype=int)

        # Related to making restart files and checking for soft exit.
        self.restart_file = restart_file
        self.make_restart_frequency = make_restart_frequency
        self.compress_restart = compress_restart
        self.soft_exit_check_frequency = soft_exit_check_frequency
        self.global_steps_since_last = {}

        self.max_histogram_size = max_histogram_size
        self.histogram_dump_file_prefix = histogram_dump_file_prefix
        self.track_histogram_size = track_histogram_size

        # Histogram initialization.
        if starting_histogram is None:
            if self.keep_histogram:
                self.histogram = SortedList()
            else:
                self.histogram = None
        else:
            self.histogram = starting_histogram

        self.delete_temp_data = delete_temp_data

        self.init_cur_tps(init_egcs)

    def init_randomized_change_params(self, randomized_change_params=None):
        if randomized_change_params is not None:
            self.randomized_change_params = randomized_change_params
            if "forbidden_bonds" in self.randomized_change_params:
                self.randomized_change_params["forbidden_bonds"] = tidy_forbidden_bonds(
                    self.randomized_change_params["forbidden_bonds"]
                )
        self.used_randomized_change_params = deepcopy(self.randomized_change_params)

        if self.no_exploration:
            if self.no_exploration_smove_adjust:
                self.used_randomized_change_params[
                    "restricted_tps"
                ] = self.restricted_tps

    def init_cur_tps(self, init_egcs=None):
        """
        Set current positions of self's trajectory from init_egcs while checking that the resulting trajectory points are valid.
        init_egcs : a list of ExtGraphCompound objects to be set as positions; if None the procedure terminates without doing anything.
        """
        if init_egcs is None:
            return
        self.cur_tps = []
        for egc in init_egcs:
            if egc_valid_wrt_change_params(egc, **self.used_randomized_change_params):
                added_tp = TrajectoryPoint(egc=egc)
            else:
                raise InvalidStartingMolecules
            if self.no_exploration:
                if added_tp not in self.restricted_tps:
                    raise InvalidStartingMolecules
            added_tp = self.hist_checked_tps([added_tp])[0]
            if self.min_function is not None:
                # Initialize the minimized function's value in the new trajectory point and check that it is not None
                cur_min_func_val = self.eval_min_func(added_tp)
                if cur_min_func_val is None:
                    raise InvalidStartingMolecules
            self.cur_tps.append(added_tp)
        if self.num_saved_candidates is not None:
            self.update_saved_candidates()
        self.update_histogram(list(range(self.num_replicas)))
        self.update_global_histogram()

    # Acceptance rejection rules.
    def accept_reject_move(self, new_tps, prob_balance, replica_ids=[0]):
        self.MC_step_counter += 1

        accepted = self.acceptance_rule(new_tps, prob_balance, replica_ids=replica_ids)
        if accepted:
            for new_tp, replica_id in zip(new_tps, replica_ids):
                self.cur_tps[replica_id] = new_tp
                self.moves_since_changed[replica_id] = 0
        else:
            for new_tp, replica_id in zip(new_tps, replica_ids):
                self.moves_since_changed[replica_id] += 1
                if self.keep_histogram and self.histogram_save_rejected:
                    tp_in_histogram = new_tp in self.histogram
                    if tp_in_histogram:
                        tp_index = self.histogram.index(new_tp)
                        new_tp.copy_extra_data_to(self.histogram[tp_index])
                    else:
                        self.histogram.add(new_tp)

        if self.num_saved_candidates is not None:
            self.update_saved_candidates()
        if self.keep_histogram:
            self.update_histogram(replica_ids)
            if self.max_histogram_size is not None:
                if len(self.histogram) > self.max_histogram_size:
                    self.histogram2pkl()

        return accepted

    def acceptance_rule(self, new_tps, prob_balance, replica_ids=[0]):

        if self.no_exploration:
            for new_tp in new_tps:
                if new_tp not in self.restricted_tps:
                    return False
        new_tot_pot_vals = [
            self.tot_pot(new_tp, replica_id)
            for new_tp, replica_id in zip(new_tps, replica_ids)
        ]

        # Check we have not created any invalid molecules.
        if None in new_tot_pot_vals:
            return False

        prev_tot_pot_vals = [
            self.tot_pot(self.cur_tps[replica_id], replica_id)
            for replica_id in replica_ids
        ]

        if (self.betas is not None) and self.virtual_beta_present(replica_ids):
            vnew_tot_pot_vals = []
            vprev_tot_pot_vals = []
            for replica_id, new_tot_pot_val, prev_tot_pot_val in zip(
                replica_ids, new_tot_pot_vals, prev_tot_pot_vals
            ):
                if self.virtual_beta_id(replica_id):
                    vnew_tot_pot_vals.append(new_tot_pot_val)
                    vprev_tot_pot_vals.append(prev_tot_pot_val)
            return min(vnew_tot_pot_vals) <= min(vprev_tot_pot_vals)

        delta_pot = prob_balance + sum(new_tot_pot_vals) - sum(prev_tot_pot_vals)

        if delta_pot <= 0.0:
            return True
        else:
            return random.random() < exp_wexceptions(-delta_pot)

    def virtual_beta_present(self, beta_ids):
        return any([self.virtual_beta_id(beta_id) for beta_id in beta_ids])

    def virtual_beta_id(self, beta_id):
        if self.betas is None:
            return False
        else:
            return self.betas[beta_id] is None

    def tot_pot(self, tp, replica_id, init_bias=0.0):
        """
        Total potential including minimized function, constraining and biasing potentials.
        """
        tot_pot = init_bias
        if self.min_function is not None:
            min_func_val = self.eval_min_func(tp)
            if min_func_val is None:
                return None
            if self.betas[replica_id] is not None:
                min_func_val *= self.betas[replica_id]
            tot_pot += min_func_val
        tot_pot += self.biasing_potential(tp, replica_id)
        if self.bound_enforcing_coeff is not None:
            tot_pot += self.bound_enforcing_pot(tp, replica_id)
        return tot_pot

    def bound_enforcing_pot(self, tp, replica_id):
        return self.bound_enforcing_coeff * self.bound_outlie(tp.egc, replica_id)

    def bound_outlie(self, egc, replica_id):
        output = 0
        if self.hydrogen_nums is not None:
            if egc.chemgraph.tot_nhydrogens() != self.hydrogen_nums[replica_id]:
                output += abs(
                    egc.chemgraph.tot_nhydrogens() - self.hydrogen_nums[replica_id]
                )
        if egc.chemgraph.num_connected != 1:
            output += egc.chemgraph.num_connected() - 1
        if "final_nhatoms_range" in self.randomized_change_params:
            final_nhatoms_range = self.randomized_change_params["final_nhatoms_range"]
            if egc.num_heavy_atoms() > final_nhatoms_range[1]:
                output += egc.num_heavy_atoms() - final_nhatoms_range[1]
            else:
                if egc.num_heavy_atoms() < final_nhatoms_range[0]:
                    output += final_nhatoms_range[0] - egc.num_heavy_atoms()
        return output

    def min_over_virtual(self, tot_pot_vals, replica_ids):
        output = None
        for tot_pot_val, replica_id in zip(tot_pot_vals, replica_ids):
            if (
                self.virtual_beta_id(replica_id)
                and (output is not None)
                and (output > tot_pot_val)
            ):
                output = tot_pot_val
        return output

    # TODO do we still need Metropolis_rejection_prob? Does not appear anywhere.
    def tp_pair_order_prob(
        self, replica_ids, tp_pair=None, Metropolis_rejection_prob=False
    ):
        if tp_pair is None:
            tp_pair = [self.cur_tps[replica_id] for replica_id in replica_ids]
        cur_tot_pot_vals = [
            self.tot_pot(tp, replica_id) for tp, replica_id in zip(tp_pair, replica_ids)
        ]
        if None in cur_tot_pot_vals:
            return None
        switched_tot_pot_vals = [
            self.tot_pot(tp, replica_id)
            for tp, replica_id in zip(tp_pair, replica_ids[::-1])
        ]
        if self.virtual_beta_present(replica_ids):
            if all([(self.betas[replica_id] is None) for replica_id in replica_ids]):
                return 0.5
            else:
                cur_virt_min = self.min_over_virtual(cur_tot_pot_vals, replica_ids)
                switched_virt_min = self.min_over_virtual(
                    switched_tot_pot_vals, replica_ids
                )
                if cur_virt_min == switched_virt_min:
                    return 0.5
                if cur_virt_min < switched_virt_min:
                    return 1.0
                else:
                    return None
        else:
            delta_pot = sum(cur_tot_pot_vals) - sum(switched_tot_pot_vals)
            if Metropolis_rejection_prob:
                if delta_pot < 0.0:
                    return 0.0
                else:
                    return 1.0 - exp_wexceptions(-delta_pot)
            else:
                exp_val = exp_wexceptions(-delta_pot)
                if np.isinf(exp_val):
                    return None
                else:
                    try:
                        return (1.0 + exp_val) ** (-1)
                    except FloatingPointError:
                        return None

    # Basic move procedures.
    def MC_step(self, replica_id=0, **dummy_kwargs):
        changed_tp = self.cur_tps[replica_id]
        changed_tp.init_possibility_info(**self.used_randomized_change_params)
        new_tp, prob_balance = randomized_change(
            changed_tp,
            visited_tp_list=self.histogram,
            **self.used_randomized_change_params
        )
        if new_tp is None:
            return False
        new_tps = self.hist_checked_tps([new_tp])
        accepted = self.accept_reject_move(
            new_tps, prob_balance, replica_ids=[replica_id]
        )
        return accepted

    def genetic_MC_step(self, replica_ids):
        old_cg_pair = [
            self.cur_tps[replica_id].egc.chemgraph for replica_id in replica_ids
        ]
        new_cg_pair, prob_balance = randomized_cross_coupling(
            old_cg_pair,
            visited_tp_list=self.histogram,
            **self.used_randomized_change_params
        )
        self.num_attempted_cross_couplings += 1
        if new_cg_pair is not None:
            if "nhatoms_range" in self.used_randomized_change_params:
                nhatoms_range = self.used_randomized_change_params["nhatoms_range"]
                invalid = False
                for nm in new_cg_pair:
                    invalid = (nm.nhatoms() < nhatoms_range[0]) or (
                        nm.nhatoms() > nhatoms_range[1]
                    )
                    if invalid:
                        break
                if invalid:
                    new_cg_pair = None
        if new_cg_pair is None:
            return False
        new_pair_tps = self.hist_checked_tps(
            [TrajectoryPoint(cg=new_cg) for new_cg in new_cg_pair]
        )
        if self.betas is not None:
            new_pair_shuffle_prob = self.tp_pair_order_prob(
                replica_ids, tp_pair=new_pair_tps
            )
            if (
                new_pair_shuffle_prob is None
            ):  # minimization function values for new_pair_tps have None and are thus invalid.
                return False
            if random.random() > new_pair_shuffle_prob:  # shuffle
                new_pair_shuffle_prob = 1.0 - new_pair_shuffle_prob
                new_pair_tps = new_pair_tps[::-1]
            old_pair_shuffle_prob = self.tp_pair_order_prob(replica_ids)
            if old_pair_shuffle_prob is None:
                return True
            prob_balance += np.log(new_pair_shuffle_prob / old_pair_shuffle_prob)
        accepted = self.accept_reject_move(
            new_pair_tps, prob_balance, replica_ids=replica_ids
        )
        if accepted:
            self.num_accepted_cross_couplings += 1
        return accepted

    # Procedures for changing entire list of Trajectory Points at once.
    def MC_step_all(self, **mc_step_kwargs):
        output = []
        for replica_id in range(self.num_replicas):
            output.append(self.MC_step(**mc_step_kwargs, replica_id=replica_id))
        return output

    def random_changed_replica_pair(self):
        return random.sample(range(self.num_replicas), 2)

    def genetic_MC_step_all(
        self, num_genetic_tries=1, randomized_change_params=None, **dummy_kwargs
    ):
        self.init_randomized_change_params(
            randomized_change_params=randomized_change_params
        )
        for _ in range(num_genetic_tries):
            changed_replica_ids = self.random_changed_replica_pair()
            # The swap before is to avoid situations where the pair's
            # initial ordering's probability is 0 in genetic move,
            # the second is for detailed balance concerns.
            self.parallel_tempering_swap(changed_replica_ids)
            self.genetic_MC_step(changed_replica_ids)
            self.parallel_tempering_swap(changed_replica_ids)

    def parallel_tempering_swap(self, replica_ids):
        trial_tps = [
            deepcopy(self.cur_tps[replica_ids[1]]),
            deepcopy(self.cur_tps[replica_ids[0]]),
        ]
        return self.accept_reject_move(trial_tps, 0.0, replica_ids=replica_ids)

    def parallel_tempering(self, num_parallel_tempering_tries=1, **dummy_kwargs):
        if self.min_function is not None:
            for _ in range(num_parallel_tempering_tries):
                replica_ids = self.random_changed_replica_pair()
                _ = self.parallel_tempering_swap(replica_ids)

    def global_change_dict(self):
        return {
            "simple": self.MC_step_all,
            "genetic": self.genetic_MC_step_all,
            "tempering": self.parallel_tempering,
        }

    def global_random_change(
        self,
        prob_dict={"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
        **other_kwargs
    ):

        self.global_MC_step_counter += 1

        cur_procedure = random.choices(
            list(prob_dict), weights=list(prob_dict.values())
        )[0]

        global_change_dict = self.global_change_dict()
        if cur_procedure in global_change_dict:
            global_change_dict[cur_procedure](**other_kwargs)
        else:
            raise Exception("Unknown option picked.")
        if self.keep_histogram and self.track_histogram_size:
            print(
                "HIST SIZE:",
                self.global_MC_step_counter,
                cur_procedure,
                len(self.histogram),
            )

        if self.make_restart_frequency is not None:
            self.check_make_restart()
        if self.soft_exit_check_frequency is not None:
            self.check_soft_exit()
        self.update_global_histogram()

    # For convenient interfacing with other scripts.
    def convert_to_current_egcs(self, mol_in_list, mol_format=None):
        self.cur_tps = [
            TrajectoryPoint(egc=conv_func_mol_egc(mol_in, mol_format=mol_format))
            for mol_in in mol_in_list
        ]
        self.num_replicas = len(self.cur_tps)
        if self.betas is not None:
            assert len(self.betas) == self.num_replicas

    def converted_current_egcs(self, mol_format=None):
        return [
            conv_func_mol_egc(cur_tp.egc, mol_format=mol_format, forward=False)
            for cur_tp in self.cur_tps
        ]

    def change_molecule(self, mol_in, **change_molecule_params):
        return self.change_molecule_list([mol_in], **change_molecule_params)[0]

    def change_molecule_list(
        self, mol_in_list, randomized_change_params=None, mol_format=None
    ):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.init_randomized_change_params(
            randomized_change_params=randomized_change_params
        )
        _ = self.MC_step_all()
        return self.converted_current_egcs(mol_format=mol_format)

    def parallel_tempering_molecule_list(
        self, mol_in_list, mol_format=None, **other_kwargs
    ):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.parallel_tempering(**other_kwargs)
        return self.converted_current_egcs(mol_format=mol_format)

    def genetic_change_molecule_list(
        self, mol_in_list, mol_format=None, **other_kwargs
    ):
        self.convert_to_current_egcs(mol_in_list, mol_format=mol_format)
        self.genetic_MC_step_all(**other_kwargs)
        return self.converted_current_egcs(mol_format=mol_format)

    # Either evaluate minimized function or look it up.
    def eval_min_func(self, tp):
        output = tp.calc_or_lookup(self.min_function_dict)[self.min_function_name]

        # If we are keeping track of the histogram make sure all calculated data is saved there.
        # TODO combine things with update_histogram?
        if self.keep_histogram:
            tp_in_histogram = tp in self.histogram
            if tp_in_histogram:
                tp_index = self.histogram.index(tp)
                tp.copy_extra_data_to(self.histogram[tp_index])
            else:
                self.histogram.add(deepcopy(tp))

        if self.delete_temp_data is not None:
            for dtd_identifier in self.delete_temp_data:
                if dtd_identifier in tp.calculated_data:
                    del tp.calculated_data[dtd_identifier]
        return output

    # If we want to re-merge two different trajectories.
    def combine_with(self, other_rw):
        for tp in other_rw.histogram:
            if tp in self.histogram:
                tp_index = self.histogram.index(tp)
                self.histogram[tp_index].num_visits += tp.num_visits
            else:
                self.histogram.add(tp)

    def biasing_potential(self, tp, replica_id):
        cur_beta_virtual = self.virtual_beta_id(replica_id)
        if cur_beta_virtual:
            bias_coeff = self.vbeta_bias_coeff
        else:
            bias_coeff = self.bias_coeff

        if (bias_coeff is None) or (tp.num_visits is None):
            return 0.0
        else:
            if (self.histogram is None) or (tp not in self.histogram):
                return 0.0
            tp_index = self.histogram.index(tp)
            if self.bias_pot_all_replicas:
                cur_visit_num = 0
                for other_replica_id in range(self.num_replicas):
                    if cur_beta_virtual == self.virtual_beta_id(other_replica_id):
                        cur_visit_num += self.histogram[tp_index].num_visits[
                            other_replica_id
                        ]
                cur_visit_num = float(cur_visit_num)
            else:
                cur_visit_num = self.histogram[tp_index].num_visits[replica_id]
            return cur_visit_num * bias_coeff

    def update_histogram(self, replica_ids):
        if self.keep_histogram:
            for replica_id in replica_ids:
                cur_tp = self.cur_tps[replica_id]
                tp_in_hist = cur_tp in self.histogram
                if not tp_in_hist:
                    self.histogram.add(deepcopy(cur_tp))
                cur_tp_index = self.histogram.index(cur_tp)
                if tp_in_hist:
                    cur_tp.copy_extra_data_to(
                        self.histogram[cur_tp_index], linear_storage=self.linear_storage
                    )
                else:
                    if self.linear_storage:
                        self.histogram[cur_tp_index].clear_possibility_info()
                        self.histogram[
                            cur_tp_index
                        ].egc.chemgraph.pair_equivalence_matrix = None

                if self.histogram[cur_tp_index].first_MC_step_encounter is None:
                    self.histogram[
                        cur_tp_index
                    ].first_MC_step_encounter = self.MC_step_counter

                if self.histogram[cur_tp_index].first_global_MC_step_encounter is None:
                    self.histogram[
                        cur_tp_index
                    ].first_global_MC_step_encounter = self.global_MC_step_counter

                if self.keep_full_trajectory:
                    if self.histogram[cur_tp_index].visit_step_ids is None:
                        self.histogram[cur_tp_index].visit_step_ids = [
                            [] for _ in range(self.num_replicas)
                        ]
                    self.histogram[cur_tp_index].visit_step_ids[replica_id].append(
                        self.MC_step_counter
                    )
                if self.visit_num_count_acceptance:
                    self.update_num_visits(cur_tp_index, replica_id)

    def update_global_histogram(self):
        if self.keep_histogram:
            for replica_id, cur_tp in enumerate(self.cur_tps):
                cur_tp_index = self.histogram.index(cur_tp)
                if self.keep_full_trajectory:
                    if self.histogram[cur_tp_index].visit_global_step_ids is None:
                        self.histogram[cur_tp_index].visit_global_step_ids = [
                            [] for _ in range(self.num_replicas)
                        ]
                    self.histogram[cur_tp_index].visit_global_step_ids[
                        replica_id
                    ].append(self.global_MC_step_counter)
                if not self.visit_num_count_acceptance:
                    self.update_num_visits(cur_tp_index, replica_id)

    def update_num_visits(self, tp_index, replica_id):
        if self.histogram[tp_index].num_visits is None:
            self.histogram[tp_index].num_visits = np.zeros(
                (self.num_replicas,), dtype=int
            )
        self.histogram[tp_index].num_visits[replica_id] += 1

    def hist_checked_tps(self, tp_list):
        """
        Return a version of tp_list where all entries are replaced with references to self.histogram.
        tp_list : list of TrajectoryPoint objects
        """
        if self.histogram is None:
            return tp_list
        output = []
        for tp in tp_list:
            if tp in self.histogram:
                output.append(deepcopy(self.histogram[self.histogram.index(tp)]))
            else:
                output.append(tp)
        return output

    def clear_histogram_visit_data(self):
        for tp in self.histogram:
            tp.num_visits = None
            if self.keep_full_trajectory:
                tp.visit_step_ids = None

    def update_saved_candidates(self):
        for tp in self.cur_tps:
            if self.min_function_name in tp.calculated_data:
                new_cc = CandidateCompound(
                    tp, tp.calculated_data[self.min_function_name]
                )
                if new_cc not in self.saved_candidates:
                    self.saved_candidates.append(new_cc)
        self.saved_candidates.sort()
        if len(self.saved_candidates) > self.num_saved_candidates:
            del self.saved_candidates[self.num_saved_candidates :]

    # Some properties for more convenient trajectory analysis.
    def ordered_trajectory(self):
        return ordered_trajectory(
            self.histogram,
            global_MC_step_counter=self.global_MC_step_counter,
            num_replicas=self.num_replicas,
        )

    def ordered_trajectory_ids(self):
        assert self.keep_full_trajectory
        return ordered_trajectory_ids(
            self.histogram,
            global_MC_step_counter=self.global_MC_step_counter,
            num_replicas=self.num_replicas,
        )

    # Quality-of-life-related.
    def frequency_checker(self, identifier, frequency):
        if identifier not in self.global_steps_since_last:
            self.global_steps_since_last[identifier] = 1
        output = self.global_steps_since_last[identifier] == frequency
        if output:
            self.global_steps_since_last[identifier] = 1
        else:
            self.global_steps_since_last[identifier] += 1
        return output

    def check_make_restart(self):
        if self.frequency_checker("make_restart", self.make_restart_frequency):
            self.make_restart()

    def check_soft_exit(self):
        if self.frequency_checker("soft_exit", self.soft_exit_check_frequency):
            if os.path.isfile("EXIT"):
                self.make_restart()
                raise SoftExitCalled

    def make_restart(
        self, restart_file: str or None = None, tarball: bool or None = None
    ):
        """
        Create a file containing all information needed to restart the simulation from the current point.
        restart_file : name of the file where the dump is created; if None self.restart_file is used
        """
        if restart_file is None:
            restart_file = self.restart_file
        saved_data = {
            "cur_tps": self.cur_tps,
            "MC_step_counter": self.MC_step_counter,
            "global_MC_step_counter": self.global_MC_step_counter,
            "num_attempted_cross_couplings": self.num_attempted_cross_couplings,
            "num_accepted_cross_couplings": self.num_accepted_cross_couplings,
            "moves_since_changed": self.moves_since_changed,
            "global_steps_since_last": self.global_steps_since_last,
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "min_function_name": self.min_function_name,
            "betas": self.betas,
        }
        if self.keep_histogram:
            saved_data = {**saved_data, "histogram": self.histogram}
        if self.num_saved_candidates is not None:
            saved_data = {**saved_data, "saved_candidates": self.saved_candidates}
        if tarball is None:
            tarball = self.compress_restart
        if tarball:
            dump2tar(saved_data, restart_file)
        else:
            dump2pkl(saved_data, restart_file)

    def restart_from(self, restart_file: str or None = None):
        """
        Recover all data from
        restart_file : name of the file from which the data is recovered; if None self.restart_file is used
        """
        if restart_file is None:
            restart_file = self.restart_file
        recovered_data = loadpkl(restart_file)
        self.cur_tps = recovered_data["cur_tps"]
        self.MC_step_counter = recovered_data["MC_step_counter"]
        self.global_MC_step_counter = recovered_data["global_MC_step_counter"]
        self.num_attempted_cross_couplings = recovered_data[
            "num_attempted_cross_couplings"
        ]
        self.num_accepted_cross_couplings = recovered_data[
            "num_accepted_cross_couplings"
        ]
        self.moves_since_changed = recovered_data["moves_since_changed"]
        self.global_steps_since_last = recovered_data["global_steps_since_last"]
        if self.keep_histogram:
            self.histogram = recovered_data["histogram"]
        if self.num_saved_candidates is not None:
            self.saved_candidates = recovered_data["saved_candidates"]
        np.random.set_state(recovered_data["numpy_rng_state"])
        random.setstate(recovered_data["random_rng_state"])

    def histogram2pkl(self):
        """
        Dump a histogram to a dump file with a yet unoccupied name.
        """
        dump_id = 1
        dump_name = self.histogram_pkl_dump(dump_id)
        while os.path.isfile(dump_name):
            dump_id += 1
            dump_name = self.histogram_pkl_dump(dump_id)
        dump2pkl(self.histogram, dump_name)
        self.cur_tps = deepcopy(self.cur_tps)
        if self.num_saved_candidates is not None:
            self.saved_candidates = deepcopy(self.saved_candidates)
        self.histogram.clear()
        self.cur_tps = self.hist_checked_tps(self.cur_tps)

    def histogram_pkl_dump(self, dump_id: int):
        """
        Returns name of the histogram dump file for a given dump_id.
        dump_id : int id of the dump
        """
        return self.histogram_dump_file_prefix + str(dump_id) + ".pkl"


def merge_random_walks(*rw_list):
    output = rw_list[0]
    for rw in rw_list[1:]:
        output.combine_trajectory(rw)
    return output


# Some procedures for convenient RandomWalk analysis.
def histogram_num_replicas(histogram):
    for tp in histogram:
        if tp.visit_step_ids is not None:
            return len(tp.visit_step_ids)
    return None


def ordered_trajectory_ids(histogram, global_MC_step_counter=None, num_replicas=None):
    if num_replicas is None:
        num_replicas = histogram_num_replicas(histogram)
        if num_replicas is None:
            raise DataUnavailable
    if global_MC_step_counter is None:
        global_MC_step_counter = 0
        for tp in histogram:
            if tp.visit_global_step_ids is not None:
                for replica_visits in tp.visit_global_step_ids:
                    if len(replica_visits) != 0:
                        global_MC_step_counter = max(
                            global_MC_step_counter, max(replica_visits)
                        )
    output = np.zeros((global_MC_step_counter + 1, num_replicas), dtype=int)
    output[:, :] = -1
    for tp_id, tp in enumerate(histogram):
        if tp.visit_global_step_ids is not None:
            for replica_id, replica_visits in enumerate(tp.visit_global_step_ids):
                for replica_visit in replica_visits:
                    output[replica_visit, replica_id] = tp_id
    for step_id in range(global_MC_step_counter):
        true_step_id = step_id + 1
        for replica_id in range(num_replicas):
            if output[true_step_id, replica_id] == -1:
                output[true_step_id, replica_id] = output[true_step_id - 1, replica_id]
    return output


def ordered_trajectory(histogram, **ordered_trajectory_ids_kwargs):
    output = []
    for tp_ids in ordered_trajectory_ids(histogram, **ordered_trajectory_ids_kwargs):
        output.append([histogram[tp_id] for tp_id in tp_ids])
    return output


def average_wait_number(histogram):
    return average_wait_number_from_traj_ids(ordered_trajectory_ids(histogram))


def average_wait_number_from_traj_ids(traj_ids):
    num_replicas = traj_ids.shape[-1]
    output = np.zeros((num_replicas,), dtype=int)
    cur_time = np.zeros((num_replicas,), dtype=int)
    prev_ids = traj_ids[0]
    for cur_ids in traj_ids[1:]:
        for counter, (cur_id, prev_id) in enumerate(zip(cur_ids, prev_ids)):
            if cur_id == prev_id:
                cur_time[counter] += 1
            else:
                cur_time[counter] = 0
        output[:] += cur_time[:]
        prev_ids = cur_ids

    return output / traj_ids.shape[0]
