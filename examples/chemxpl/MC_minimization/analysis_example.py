from bmapqml.chemxpl.plotting import Analyze

ana = Analyze(path="/store/jan/trash/plotting_data/restart_files/restart_files/", verbose=False)

ALL_HISTOGRAMS, GLOBAL_HISTOGRAM,ALL_TRAJECTORIES = ana.parse_results()

ana.pareto_plot(GLOBAL_HISTOGRAM, ana.pareto(GLOBAL_HISTOGRAM), ALL_PARETOS=ALL_HISTOGRAMS)
ana.result_spread(ALL_HISTOGRAMS, label="xTB_MMFF_electrolyte")
#remember 8 different trajectories and 8 different temperatures
#first index is the seed, second index is the temperature
ana.trajectory_plot(ALL_TRAJECTORIES[0][0])