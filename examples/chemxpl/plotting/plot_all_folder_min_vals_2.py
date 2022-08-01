import sys, os, glob, math
import matplotlib.pyplot as plt

colors = ["blue", "red", "green", "yellow"]

quant_id = {"min_func_val": 1, "hist_size": 2}

for quant in ["min_func_val", "hist_size"]:
    for folder_id, folder in enumerate(sys.argv[1:]):
        cur_color = colors[folder_id]
        files = glob.glob(folder + "/*.expl_data")
        files.sort()
        legend = folder
        for f_id, f in enumerate(files):
            if f_id == 1:
                legend = None
            inp_f = open(f, "r")
            lines = inp_f.readlines()
            inp_f.close()
            x_vals = []
            y_vals = []
            for l in lines[1:]:
                l_spl = l.split()
                x_vals.append(math.log(float(l_spl[0])))
                y_val = float(l_spl[quant_id[quant]])
                if quant == "hist_size":
                    y_val = math.log(y_val)
                y_vals.append(y_val)
            plt.plot(x_vals, y_vals, label=legend, color=cur_color, linestyle="dashed")
    plt.legend()
    plt.savefig(quant + ".png")
    plt.clf()
