import matplotlib.pyplot as plt
from bmapqml.chemxpl.plotting import Analyze

ana = Analyze(path="/store/jan/trash/plotting_data/restart_files/")
HISTOGRAM, TRAJECTORY = ana.parse_results()
print(HISTOGRAM)
plt.plot(TRAJECTORY["xTB_MMFF_electrolyte"])
plt.show()