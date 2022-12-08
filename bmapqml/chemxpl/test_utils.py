# Utils that mainly appear in tests or analysis of data.
from ..data import NUCLEAR_CHARGE
from sortedcontainers import SortedDict
from .random_walk import (
    randomized_change,
    TrajectoryPoint,
    RandomWalk,
    full_change_list,
    minimized_change_list,
    random_choice_from_nested_dict,
    egc_change_func,
    inverse_procedure,
)
import numpy as np
import copy


def elements_in_egc_list(egc_list, as_elements=True):
    nuclear_charges = []
    for egc in egc_list:
        for nc in egc.true_ncharges():
            if nc not in nuclear_charges:
                nuclear_charges.append(nc)
    nuclear_charges.sort()
    if as_elements:
        output = []
        for el, nc in NUCLEAR_CHARGE.items():
            if nc in nuclear_charges:
                output.append(el)
    else:
        output = nuclear_charges
    return output


def egc_list_sizes(egc_list):
    sizes = []
    for egc in egc_list:
        sizes.append((egc.num_heavy_atoms(), egc.num_atoms()))
    return sizes


def egc_list_nhatoms_hist(egc_list):
    histogram = {}
    for egc in egc_list:
        nhatoms = egc.num_heavy_atoms()
        if nhatoms not in histogram:
            histogram[nhatoms] = 0
        histogram[nhatoms] += 1
    return SortedDict(histogram)


# For numerically verifying that detailed balance is satisfied.
def genetic_move_attempt(tp_init, **randomized_change_params):
    rw = RandomWalk(init_egcs=[tp.egc for tp in tp_init], **randomized_change_params)
    pair, prob_balance = rw.trial_genetic_MC_step([0, 1])
    if pair is None:
        tpair = None
    else:
        tpair = tuple(pair)
    return tpair, prob_balance


trial_attempt_funcs = {TrajectoryPoint: randomized_change, tuple: genetic_move_attempt}


def calc_bin_id(x, bin_size=None):
    if bin_size is None:
        return 0
    if np.abs(x) < 0.5 * bin_size:
        return 0
    output = int(x / bin_size - 0.5)
    if x < 0.0:
        output *= -1
    return output


def check_one_sided_prop_probability(
    tp_init, tp_trial, num_attempts=10000, bin_size=None, **randomized_change_params
):
    if isinstance(tp_trial, list):
        true_list = tp_trial
    else:
        true_list = [tp_trial]
    attempt_func = trial_attempt_funcs[type(tp_init)]

    est_balances = [[] for _ in true_list]
    probs = np.zeros((len(true_list),))
    for _ in range(num_attempts):
        tp_new, prob_balance = attempt_func(tp_init, **randomized_change_params)
        if tp_new is None:
            continue
        if tp_new not in true_list:
            continue
        i = true_list.index(tp_new)
        est_balances[i].append(prob_balance)
        probs[i] += 1.0

    output = []
    for prob, est_bal in zip(probs, est_balances):
        trial_prob_arrays = {}
        observed_probs = {}
        for bal in est_bal:
            bin_id = calc_bin_id(bal, bin_size)
            if bin_id not in trial_prob_arrays:
                trial_prob_arrays[bin_id] = []
                observed_probs[bin_id] = 0.0
            trial_prob_arrays[bin_id].append(bal)
            observed_probs[bin_id] += 1.0
        trial_prob_averages = {}
        for bin_id, prob_arr in trial_prob_arrays.items():
            trial_prob_averages[bin_id] = (np.mean(prob_arr), np.std(prob_arr))
            observed_probs[bin_id] /= num_attempts
        output.append((trial_prob_averages, observed_probs))

    if isinstance(tp_trial, list):
        return output
    else:
        return output[0]


def check_prop_probability(tp1, tp2_list, label_dict=None, **one_sided_kwargs):
    """
    Check that simple MC moves satisfy detailed balance for a pair of trajectory point objects.
    """
    if isinstance(tp2_list, list):
        true_list = tp2_list
    else:
        true_list = [tp2_list]
    print("INITIAL MOLECULE:", tp1)
    forward_results = check_one_sided_prop_probability(
        tp1, true_list, **one_sided_kwargs
    )
    for tp2, (forward_trial_prob_averaged, forward_observed_probs) in zip(
        true_list, forward_results
    ):
        print("CASE:", tp2)
        if label_dict is not None:
            print("CASE LABEL:", label_dict[str(tp2)])
        (
            inverse_trial_prob_averaged,
            inverse_observed_probs,
        ) = check_one_sided_prop_probability(tp2, tp1, **one_sided_kwargs)
        hist_ids = list(inverse_observed_probs.keys())
        for forward_hist_id in forward_observed_probs.keys():
            inverted_fhi = -forward_hist_id
            if inverted_fhi not in hist_ids:
                hist_ids.append(inverted_fhi)
        for hist_id in hist_ids:
            print("BIN ID:", hist_id)
            inverted_fhi = -hist_id
            forward_present = inverted_fhi in forward_trial_prob_averaged
            inverse_present = hist_id in inverse_trial_prob_averaged
            if inverse_present:
                inverse_prob = inverse_observed_probs[hist_id]
                print("INVERSE:", *inverse_trial_prob_averaged[hist_id], inverse_prob)
            else:
                print("NO INVERSE STEPS")
            if forward_present:
                forward_prob = forward_observed_probs[hist_id]
                print("FORWARD:", *forward_trial_prob_averaged[hist_id], forward_prob)
            else:
                print("NO FORWARD STEPS")
            if forward_present and inverse_present:
                print("OBSERVED RATIO:", np.log(forward_prob / inverse_prob))


def generate_proc_example(tp, change_procedure, print_dicts=False, **other_kwargs):
    tp_copy = copy.deepcopy(tp)
    tp_copy.init_possibility_info(change_prob_dict=[change_procedure], **other_kwargs)
    tp_copy.modified_possibility_dict = copy.deepcopy(tp_copy.possibility_dict)
    while tp_copy.modified_possibility_dict:
        modification_path, _ = random_choice_from_nested_dict(
            tp_copy.modified_possibility_dict[change_procedure]
        )
        new_egc = egc_change_func(
            tp_copy.egc, modification_path, change_procedure, **other_kwargs
        )
        if new_egc is not None:
            tp_out = TrajectoryPoint(egc=new_egc)
            if print_dicts:
                inv_proc = inverse_procedure[change_procedure]
                tp_out.init_possibility_info(
                    change_prob_dict=[inv_proc], **other_kwargs
                )
                print("EXAMPLE FOR:", tp_copy, change_procedure)
                print("NEW TP:", tp_out)
                print("INVERSE PROC DICT:", tp_out.possibility_dict[inv_proc])
                print("FORWARD PROC DICT:", tp_copy.possibility_dict[change_procedure])
                tp_out.possibility_dict = None
            return tp_out
        tp_copy.delete_mod_path([change_procedure, *modification_path])
    return None


def generate_proc_sample_dict(
    tp_init, change_prob_dict=full_change_list, **other_kwargs
):
    l = []
    d = {}
    for change_procedure in change_prob_dict:
        tp_new = generate_proc_example(tp_init, change_procedure, **other_kwargs)
        if tp_new is not None:
            l.append(tp_new)
            d[str(tp_new)] = change_procedure
    return l, d


def all_procedure_prop_probability_checks(tp_init, num_attempts=10000, **other_kwargs):
    l, d = generate_proc_sample_dict(tp_init, **other_kwargs)
    check_prop_probability(
        tp_init, l, label_dict=d, num_attempts=num_attempts, **other_kwargs
    )
