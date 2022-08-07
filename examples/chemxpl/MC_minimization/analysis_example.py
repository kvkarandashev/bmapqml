from bmapqml.chemxpl.plotting import Analyze

ana   = Analyze("/store/jan/trash/plotting_data2/restart_examples/", verbose=True)
ALL_HISTOGRAMS, GLOBAL_HISTOGRAM,ALL_TRAJECTORIES = ana.parse_results()

ana.plot_pareto(GLOBAL_HISTOGRAM, ana.pareto(GLOBAL_HISTOGRAM), ALL_PARETOS=ALL_HISTOGRAMS)
ana.plot_result_spread(ALL_HISTOGRAMS, label="xTB_MMFF_min_en_conf_electrolyte")
#remember 8 different trajectories and 8 different temperatures
#first index is the seed, second index is the temperature
ana.plot_trajectory(ALL_TRAJECTORIES[0][7])
ana.plot_chem_space(GLOBAL_HISTOGRAM, label="xTB_MMFF_min_en_conf_electrolyte")
ana.plot_trajectory_loss(ALL_TRAJECTORIES)
