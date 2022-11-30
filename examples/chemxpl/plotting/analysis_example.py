from bmapqml.chemxpl.plotting import Analyze
import pdb as pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
#ana   = Analyze("/store/jan/trash/plotting_data2/restart_examples/", verbose=True)

ana   = Analyze("/data/jan/konstantin/xTB_dipsolv_opt_1_none_strong_dipole/data_xTB_dipsolv_opt_1_none_strong_dipole_1/results/*", verbose=True)
#this will give you dataframes with the following columns:
# - index, smiles, When it was first encountered, Dipole, gap, Dipole**2 +  gap**2
ALL_HISTOGRAMS, GLOBAL_HISTOGRAM,ALL_TRAJECTORIES = ana.parse_results()



#The one with the proper convex whose only flaw was including too many point. *too many points in the convex hull
PARETO              = ana.pareto(GLOBAL_HISTOGRAM)
#The red line from the last plot
PARETO_CORRECTED    = ana.pareto_correct(GLOBAL_HISTOGRAM)
ana.plot_pareto(GLOBAL_HISTOGRAM,PARETO,PARETO_CORRECTED) # , ALL_PARETOS=ALL_HISTOGRAMS)
pdb.set_trace()

exit()



"""
global_double = pd.concat(ALL_HISTOGRAMS)
DUPLICATES = global_double[global_double.duplicated(subset=['SMILES'], keep=False)]
#DUPLICATES[DUPLICATES['SMILES'] == "CC(O)COC(O)(F)F"]
#loss_values = DUPLICATES[DUPLICATES['SMILES'] == "CC(O)COC(O)(F)F"]["xTB_MMFF94_morfeus_electrolyte"].values
#span = abs(min(loss_values)-max(loss_values))

MAX_VARIANCE = []
for smi in tqdm(DUPLICATES["SMILES"].values):
    loss_values = DUPLICATES[DUPLICATES['SMILES'] == smi]["xTB_MMFF94_morfeus_electrolyte"].values
    span = abs(min(loss_values)-max(loss_values))
    MAX_VARIANCE.append(span)
    
MAX_VARIANCE = np.array(MAX_VARIANCE)


# find those molecules with the highest variance
print(np.mean(MAX_VARIANCE))
print(np.std(MAX_VARIANCE))
plt.hist(MAX_VARIANCE, bins=200)
plt.savefig("max_variance.png", dpi=600)

pdb.set_trace()
"""


ana.plot_result_spread(ALL_HISTOGRAMS, label="xTB_MMFF94_morfeus_electrolyte")
#remember 8 different trajectories and 8 different temperatures
#first index is the seed, second index is the temperature

#ana.plot_trajectory(ALL_TRAJECTORIES[0][7])

ana.plot_chem_space(GLOBAL_HISTOGRAM, label="xTB_MMFF94_morfeus_electrolyte")
#ana.plot_trajectory_loss(ALL_TRAJECTORIES)
ana.export_csv(GLOBAL_PARETO)



"""
global_double = pd.concat(ALL_HISTOGRAMS)
DUPLICATES = global_double[global_double.duplicated(subset=['SMILES'], keep=False)]
DUPLICATES[DUPLICATES['SMILES'] == "CC(O)COC(O)(F)F"]
loss_values = DUPLICATES[DUPLICATES['SMILES'] == "CC(O)COC(O)(F)F"]["xTB_MMFF94_morfeus_electrolyte"].values
span = abs(min(loss_values)-max(loss_values))

MAX_VARIANCE = []
for smi in DUPLICATES["SMILES"].values:
    loss_values = DUPLICATES[DUPLICATES['SMILES'] == smi]["xTB_MMFF94_morfeus_electrolyte"].values
    span = abs(min(loss_values)-max(loss_values))
    MAX_VARIANCE.append(span)
    
MAX_VARIANCE = np.array(MAX_VARIANCE)
"""