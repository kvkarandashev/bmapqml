import sys, glob, math
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8.0, 6.0))

colors = ["blue", "red", "green", "yellow"]

quant_id = {"min_func_val": 1, "hist_size": 2}

plt.rcParams.update({"font.size": 16})

axis_label_fontsize = 20
legend_fontsize = 20

last_linearity = 90000


for quant in ["min_func_val", "hist_size"]:
    for folder_id, folder_pair in enumerate(sys.argv[1:]):
        folder_pair_split = folder_pair.split(":")
        folder = folder_pair_split[1]
        cur_color = colors[folder_id]
        files = glob.glob(folder + "/*.expl_data")
        files.sort()
        legend = folder_pair_split[0].replace("_", " ")
        cur_legend = legend
        for f_id, f in enumerate(files):
            if f_id == 1:
                cur_legend = None
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
            plt.plot(
                x_vals, y_vals, label=cur_legend, color=cur_color, linestyle="dashed"
            )
        if quant == "hist_size":
            x_lin = np.array(x_vals[-last_linearity:])
            y_lin = np.array(y_vals[-last_linearity:])
            linear_model = np.polyfit(x_lin, y_lin, 1)
            print("Linear fit coefficients for ", legend, ":", linear_model)

    plt.legend(fontsize=legend_fontsize, frameon=False)
    plt.xlabel(r"$\mathrm{ln}(N_{\mathrm{MC\,\,steps}})$", fontsize=axis_label_fontsize)
    if quant == "hist_size":
        ylabel = r"$\mathrm{ln}(N_{\mathrm{histogram}})$"
    else:
        ylabel = r"best $F$ value"

    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.savefig(quant + ".png")
    plt.clf()
