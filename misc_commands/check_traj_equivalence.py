from bmapqml.utils import loadpkl
import sys


def not_match(str_in):
    print(str_in)
    quit()


restart_contents = []

for restart_file in sys.argv[1:3]:
    restart_contents.append(loadpkl(restart_file))

histograms = []

print("global_MC_step_counter values and histogram sizes:")
for rc in restart_contents:
    histograms.append(rc["histogram"])
    print(rc["global_MC_step_counter"], len(histograms[-1]))

if len(histograms[0]) != len(histograms[1]):
    not_match("Histogram lengths don't match.")

matching = True

for tp1, tp2 in zip(*histograms):
    if tp1 != tp2:
        not_match("Trajectory points don't match.")
    if (tp1.visit_step_ids is None) or (tp2.visit_step_ids is None):
        if tp1.visit_step_ids is tp2.visit_step_ids:
            continue
        else:
            not_match("Visit ids don't match.")
    for vis_ids1, vis_ids2 in zip(tp1.visit_step_ids, tp2.visit_step_ids):
        if vis_ids1 != vis_ids2:
            not_match("Visit ids don't match.")

# Check that auxiliary data stored in restart files matches.
# SHOULD be satisfied if trajectories are the same unless there is a bug in the code.
extra_data_keys = [
    "cur_tps",
    "MC_step_counter",
    "global_MC_step_counter",
    "num_attempted_cross_couplings",
    "num_valid_cross_couplings",
    "num_accepted_cross_couplings",
    "num_attempted_simple_moves",
    "num_valid_simple_moves",
    "num_accepted_simple_moves",
    "num_attempted_tempering_swaps",
    "num_accepted_tempering_swaps",
    "moves_since_changed",
    "global_steps_since_last",
]

for extra_data_key in extra_data_keys:
    comp_res = (
        restart_contents[0][extra_data_key] == restart_contents[1][extra_data_key]
    )
    if not isinstance(comp_res, bool):
        comp_res = comp_res.all()
    if not comp_res:
        not_match("Not match in extra data key", extra_data_key)
print("Saved trajectories match.")
