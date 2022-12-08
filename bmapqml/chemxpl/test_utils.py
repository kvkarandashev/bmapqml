# Utils that mainly appear in tests or analysis of data.
from ..data import NUCLEAR_CHARGE
from sortedcontainers import SortedDict
from .random_walk import (
    randomized_change,
    TrajectoryPoint,
    RandomWalk,
    full_change_list,
    random_choice_from_nested_dict,
    egc_change_func,
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


def check_one_sided_prop_probability(
    tp_init, tp_trial, num_attempts=10000, **randomized_change_params
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
        if len(est_bal) == 0:
            output.append((0.0, 0.0, prob / num_attempts))
        else:
            output.append((np.mean(est_bal), np.std(est_bal), prob / num_attempts))
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
    for tp2, (forward_est_bal_mean, forward_est_bal_std, forward_est_obs) in zip(
        true_list, forward_results
    ):
        print("CASE:", tp2)
        if label_dict is not None:
            print("CASE LABEL:", label_dict[str(tp2)])
        print("FORWARD:", forward_est_bal_mean, forward_est_bal_std)
        (
            inverse_est_bal_mean,
            inverse_est_bal_std,
            inverse_est_obs,
        ) = check_one_sided_prop_probability(tp2, tp1, **one_sided_kwargs)
        print("INVERSE:", inverse_est_bal_mean, inverse_est_bal_std)
        try:
            print(
                "OBSERVED:",
                forward_est_obs,
                inverse_est_obs,
                np.log(forward_est_obs / inverse_est_obs),
            )
        except FloatingPointError:
            print("NO INVERSE MOVES OBSERVED, FORWARD PROBABILITY:", forward_est_obs)


def generate_proc_example(tp, change_procedure, **other_kwargs):
    tp_copy = copy.deepcopy(tp)
    change_prob_dict = [change_procedure]
    tp_copy.init_possibility_info(change_prob_dict=[change_procedure], **other_kwargs)
    try:
        modification_path, _ = random_choice_from_nested_dict(
            tp_copy.possibility_dict[change_procedure]
        )
    except KeyError:
        return None
    new_egc = egc_change_func(
        tp_copy.egc, modification_path, change_procedure, **other_kwargs
    )
    return TrajectoryPoint(egc=new_egc)


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
    check_prop_probability(tp_init, l, label_dict=d, num_attempts=10000, **other_kwargs)
