# Check that probability balance is well calculated for larger molecules.
import random
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.test_utils import check_prop_probability
from bmapqml.chemxpl.modify import randomized_cross_coupling
from bmapqml.chemxpl.minimized_functions.toy_problems import ChargeSum
import sys
from bmapqml.utils import loadpkl

restart_file = sys.argv[1]

if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    seed = 1

random.seed(seed)
np.random.seed(seed)

num_random_couplings = 1

forbidden_bonds = [
    (7, 7),
    (7, 8),
    (8, 8),
    (7, 9),
    (8, 9),
    (9, 9),
    (7, 17),
    (8, 17),
    (9, 17),
    (17, 17),
    (7, 35),
    (8, 35),
    (9, 35),
    (17, 35),
    (35, 35),
    (15, 15),
    (16, 16),
]
nhatoms_range = [1, 15]

cross_coupling_kwargs = {
    "forbidden_bonds": forbidden_bonds,
    "nhatoms_range": nhatoms_range,
}

# TODO CHECK CONSTRAINT SATISFACTION; NOT_PROTONATED???

print("Loading histogram.")
histogram = loadpkl(restart_file)["histogram"]
print("Histogram loaded.")

num_tps = 2 * num_random_couplings

tps = random.sample(histogram, num_tps)

for i in range(num_tps):
    tps[i].calculated_data = {}

tp_pairs = [(tps[2 * i], tps[2 * i + 1]) for i in range(num_random_couplings)]

new_pair_lists = [[] for _ in range(num_random_couplings)]

num_new_pairs = 4

attempts_to_generate = 40000

ln2 = np.log(2.0)

# betas = [ln2, ln2]
betas = [ln2, ln2 / 2.0]

minimized_function = ChargeSum()

for tp_pair_id, tp_pair in enumerate(tp_pairs):
    for _ in range(attempts_to_generate):
        new_cg_pair, _ = randomized_cross_coupling(
            [tp.chemgraph() for tp in tp_pair], **cross_coupling_kwargs
        )
        if new_cg_pair is None:
            continue
        tnew_pair = tuple([TrajectoryPoint(cg=cg) for cg in new_cg_pair])
        if tnew_pair not in new_pair_lists[tp_pair_id]:
            new_pair_lists[tp_pair_id].append(tnew_pair)
            if len(new_pair_lists[tp_pair_id]) == num_new_pairs:
                break

num_attempts = 80000  # 80000

for tp_pair, new_pair_list in zip(tp_pairs, new_pair_lists):
    print("BETAS:", betas)
    check_prop_probability(
        tp_pair,
        new_pair_list,
        randomized_change_params=cross_coupling_kwargs,
        num_attempts=num_attempts,
        min_function=minimized_function,
        betas=betas,
        bin_size=0.01,
    )
