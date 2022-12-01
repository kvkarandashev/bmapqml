from bmapqml.chemxpl.plotting import Analyze


result_path = "/data/jan/konstantin/xTB_dipsolv_opt_1_weak_strong_dipole/data_xTB_dipsolv_opt_1_weak_strong_dipole_5"
ana   = Analyze("{}/restart_file*".format(result_path), verbose=True)
ALL_HISTOGRAMS, GLOBAL_HISTOGRAM,ALL_TRAJECTORIES = ana.parse_results()
PARETO_CORRECTED    = ana.pareto_correct(GLOBAL_HISTOGRAM)
ana.plot_pareto(hline=0.1, vline=4.0)