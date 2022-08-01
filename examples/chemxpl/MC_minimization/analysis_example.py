import matplotlib.pyplot as plt
from bmapqml.chemxpl.plotting import Analyze
import pdb

ana = Analyze(path="/store/jan/trash/plotting_data/restart_files/restart_files/", verbose=True)
ALL_HISTOGRAMS, ALL_TRAJECTORIES = ana.parse_results()

#pdb.set_trace()
print(ALL_HISTOGRAMS)
plt.plot(ALL_TRAJECTORIES[-1]["xTB_MMFF_electrolyte"], lw=1)
plt.show()
#/store/common/konst/chemxpl_related/plotting_examples