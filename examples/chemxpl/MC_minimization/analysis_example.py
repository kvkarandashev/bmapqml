from bmapqml.chemxpl.plotting import Analyze

ana = Analyze(path="/store/jan/trash/plotting_data/restart_files/restart_files/", verbose=False)

ALL_HISTOGRAMS, GLOBAL_HISTOGRAM,ALL_TRAJECTORIES = ana.parse_results()
ana.pareto_plot(GLOBAL_HISTOGRAM, ana.pareto(GLOBAL_HISTOGRAM))
ana.result_spread(ALL_HISTOGRAMS, label="xTB_MMFF_electrolyte")