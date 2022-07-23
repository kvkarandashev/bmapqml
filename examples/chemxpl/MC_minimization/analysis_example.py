"""
Script to analyze the results of the random walk.
Currently supported are the modes:
- Atomization energy: atomization
- Band gap: gap
- Both together: atomization_gap

Creates a plot of the results in form of a pareto plot and PCA plot.

Example usage:
python analysis.py -PATH /home/jan/projects/MOLOPT/do_sim/bias/both/8/ -name both_8 -properties atomization_gap
"""

from bmapqml.chemxpl.plotting import analyze_random_walk
from bmapqml.chemxpl import rdkit_descriptors
import numpy as np
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj,sample_local_space
import argparse
import pdb

parser = argparse.ArgumentParser(description='INPUT PATH')
parser.add_argument("-PATH")
parser.add_argument("-name", default="plot")
parser.add_argument("-properties", default="atomization_gap")
args = parser.parse_args()
path = args.PATH
name=args.name


# Paths to the ML models
atomization_model, gap_model = "/store/common/jan/qm9/KRR_12000_atomization", "/store/common/jan/qm9/KRR_12000_gap"

if args.properties == "atomization_gap":
    WEIGHTS = np.array([ (1/1.9), (1/6.8)])
    min_func = multi_obj(
        [QM9_properties(atomization_model,verbose=False),QM9_properties(gap_model,verbose=False)
    ], WEIGHTS, verbose=True)

elif args.properties == "gap":
    min_func = QM9_properties(gap_model,verbose=False)

elif args.properties == "atomization":
    min_func = QM9_properties(atomization_model,verbose=False)
elif args.properties == "sample_local_space":
    target = "CCCCCCCCC"
    X_target = rdkit_descriptors.extended_get_single_FP(target, fp_type="both")
    min_func = sample_local_space(X_target, verbose=True, pot_type="harmonic",epsilon=1, sigma=10)
else:
    raise ValueError("Unknown property")

lhist = path+"QM9_histogram.pkl"
# Initialize the analysis object
ana = analyze_random_walk(lhist,trajectory=path+"QM9_trajectory.pkl", model=min_func)
# Re-evalutate the accepted molecules, save values
ana.evaluate_histogram()
#pdb.set_trace()
ana.evaluate_all_trajectory_points()
pdb.set_trace()
# Compute pareto front
ana.compute_pareto_front()
# Write pareto front to file, as well as the pareto front values and molecules
ana.write_report(name)
# Make Pareto Plot
ana.plot_pateto_front(name)
# Make PCA plot of chemical space
ana.plot_chem_space(name)