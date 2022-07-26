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

print("Saved trajectories match.")
