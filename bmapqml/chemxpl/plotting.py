from bmapqml.utils import *
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import trajectory_point_to_canonical_rdkit
from bmapqml.chemxpl.random_walk import ordered_trajectory,ordered_trajectory_from_restart
from sklearn.decomposition import PCA,TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from rdkit import RDLogger
import glob
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pdb
import seaborn as sns

lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)
import warnings

warnings.filterwarnings("ignore")

fs = 24
plt.rc("font", size=fs)
plt.rc("axes", titlesize=fs)
plt.rc("axes", labelsize=fs)
plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)
plt.rc("legend", fontsize=fs)
plt.rc("figure", titlesize=fs)


def make_pretty(axi):
    axi.spines["right"].set_color("none")
    axi.spines["top"].set_color("none")
    axi.spines["bottom"].set_position(("axes", -0.05))
    axi.spines["bottom"].set_color("black")
    axi.spines["left"].set_color("black")
    axi.yaxis.set_ticks_position("left")
    axi.xaxis.set_ticks_position("bottom")
    axi.spines["left"].set_position(("axes", -0.05))
    return axi



class Analyze:

    """
    Analysis of the results of the optimization.

    Usage:
    analysis = Analyze(path)
    analysis.parse_results()
    as shown in analysis_example.py

    path contains the path to the results of the optimization
    given as restart files in the tar format.
    """

    def __init__(self, path, quantity="dipole", full_traj=False, verbose=False):

        self.path = path
        self.results = glob.glob(path)
        self.verbose = verbose
        self.full_traj = full_traj
        self.quantity = quantity


    def parse_results(self):
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS = []
        ALL_TRAJECTORIES = []

        for run in tqdm(self.results, disable=not self.verbose):

            restart_data = loadpkl(run, compress=False)
            self.global_MC_step_counter = restart_data["global_MC_step_counter"]
            HISTOGRAM = self.to_dataframe(restart_data["histogram"])
            HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
            ALL_HISTOGRAMS.append(HISTOGRAM)
            if self.full_traj:
        
                traj = np.array(ordered_trajectory_from_restart(restart_data))
                
                CURR_TRAJECTORIES = []
                for T in range(traj.shape[1]):
                    sel_temp = traj[:, T]
                    TRAJECTORY = self.to_dataframe(sel_temp)
                    CURR_TRAJECTORIES.append(TRAJECTORY)
                ALL_TRAJECTORIES.append(CURR_TRAJECTORIES)

        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=["SMILES"])


        self.X_QUANTITY, self.GAP = (
            self.GLOBAL_HISTOGRAM["X_QUANTITY"].values,
            self.GLOBAL_HISTOGRAM["HOMO_LUMO_gap"],
        )
        self.ENCOUNTER = self.GLOBAL_HISTOGRAM["ENCONTER"].values
        self.LABELS = self.GLOBAL_HISTOGRAM.columns[1:]

        if self.full_traj:
            test_traj = ALL_TRAJECTORIES[0]
            traj_smiles = np.array([df.SMILES.values for df in test_traj])
            self.time_ordered_smiles = np.concatenate(traj_smiles.T, axis=0)
            #pdb.set_trace()
            """
            (Pdb) time_ordered_smiles[:40]
            array(['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'CF', 'N',
                'N'], dtype=object)
            """


            self.ALL_TRAJECTORIES = pd.concat(ALL_TRAJECTORIES[0])
            self.X_QUANTITY_traj, self.GAP_traj =  self.ALL_TRAJECTORIES["X_QUANTITY"].values, self.ALL_TRAJECTORIES["HOMO_LUMO_gap"]
            return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES
        else:
            return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM

    def pareto_correct(self, HISTOGRAM):
        try:
            from scipy.spatial import ConvexHull
        except:
            print("Please install scipy")
            exit()

        self.points = np.array([self.X_QUANTITY, self.GAP]).T
        self.hull = ConvexHull(self.points)
        pareto = np.unique(self.hull.simplices.flatten())
        self.PARETO = HISTOGRAM.iloc[pareto]
        self.PARETO = self.PARETO.sort_values(by=['X_QUANTITY'], ascending=False)

        
        return self.PARETO

    def pareto(self, HISTOGRAM, maxX=True, maxY=True):

        """
        Filter the histogram to keep only the pareto optimal solutions.
        """

        Xs, Ys = HISTOGRAM["X_QUANTITY"].values, HISTOGRAM["HOMO_LUMO_gap"].values
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)

        p_front = [myList[0]]

        for pair in myList[1:]:
            if maxY:
                if pair[1] >= p_front[-1][1]:
                    p_front.append(pair)
            else:
                if pair[1] <= p_front[-1][1]:
                    p_front.append(pair)

        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]

        inds = []
        for p1, p2 in zip(p_frontX, p_frontY):
            ind = 0

            for v1, v2 in zip(Xs, Ys):
                if p1 == v1 and p2 == v2:
                    inds.append(ind)
                ind += 1

        PARETO = HISTOGRAM.iloc[np.array(inds)]
        if self.verbose:
            print("Pareto optimal solutions:")
            print(PARETO)

        PARETO = PARETO.sort_values("X_QUANTITY", ascending=True)
        return PARETO

    def to_dataframe(self, obj):
        """
        Convert the trajectory point object to a dataframe
        and extract xTB values if available.
        """

        df = pd.DataFrame()

        SMILES, VALUES = self.convert_from_tps(obj)
        df["SMILES"] = SMILES
        df["ENCONTER"] = VALUES[:, 0]
        df["X_QUANTITY"] = VALUES[:, 1]
        df["HOMO_LUMO_gap"] = VALUES[:, 2]

        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def convert_from_tps(self, mols):
        """
        Convert the list of trajectory points molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """

        SMILES = []
        VALUES = []
        ENCOUNTER = []

        for tp in mols:

            curr_data = tp.calculated_data["xTB_res"]["mean"]
            
            smiles, step, x_quantity, gap = (
                trajectory_point_to_canonical_rdkit(tp, SMILES_only=True),
                tp.first_MC_step_encounter,
                curr_data[self.quantity],
                curr_data["HOMO_LUMO_gap"],
            )
            if x_quantity != None and gap != None:
                VALUES.append(
                    [
                        int(step),
                        float(x_quantity),
                        float(gap),
                    ]
                )

                ENCOUNTER.append(step)
                SMILES.append(smiles)

        SMILES, VALUES = self.make_canon(SMILES), np.array(VALUES)

        return SMILES, VALUES

    def make_canon(self, SMILES):
        """
        Convert to canonical smiles form.
        """

        CANON_SMILES = []
        for smi in SMILES:

            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            CANON_SMILES.append(can)

        return CANON_SMILES


    def plot_trajectory_loss(self, ALL_TRAJECTORIES):
        """
        Compute the loss of the trajectories.
        """

        fig, ax1 = plt.subplots(figsize=(8, 8))

        for traj in ALL_TRAJECTORIES:
            n_temps = len(traj)
            cmap = cm.coolwarm(np.linspace(0, 1, n_temps))
            for c, T in zip(cmap, range(n_temps)[:3]):

                sel_temp = traj[T]["loss"]
                N = np.arange(len(sel_temp))
                ax1.scatter(N, sel_temp, s=5, color=c, alpha=0.5)
                ax1.plot(N, sel_temp, "-", alpha=0.1)

        ax1 = make_pretty(ax1)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("loss.png", dpi=600)
        plt.close("all")

    def plot_pareto(self,name_plot, hline=None, vline=None, coloring="encounter"):
        """
        Plot the pareto optimal solutions.
        """

        fig, ax1 = plt.subplots(figsize=(8, 8))

        cc = 'gnuplot_r'
        gs = 30

        max_x = abs(vline)
        max_y = np.max(np.abs(self.GAP))

        if coloring == "encounter":
            sc=ax1.hexbin(self.X_QUANTITY/max_x,  self.GAP/max_y, gridsize=gs, mincnt=5, cmap=cc,C=self.ENCOUNTER, linewidths=0)
        if coloring == "density":
            from matplotlib import colors
            np.seterr(under='warn')
            sc=ax1.hexbin(self.X_QUANTITY_traj/max_x, self.GAP_traj /max_y, gridsize=gs,bins="log", mincnt=5,linewidths=0,norm=colors.LogNorm(vmin=100, vmax=200000))

            

        plt.xlabel(self.quantity, fontsize=21)
        plt.ylabel(
            "Gap",
            fontsize=21,
            rotation=0,
            ha="left",
            y=1.05,
            labelpad=-50,
            weight=500,
        )
        #clb = plt.colorbar(sc)
        #
        #create colorbar with name clb and ticks at the middle of the bins
        ax1 = make_pretty(ax1)
        
        if coloring == "encounter":
            clb = fig.colorbar(sc)
            clb.set_label("step encountered", fontsize=21)
            ticks =  [i*5000 for i in range(1,7)]
            clb.set_ticks(ticks)
            clb.set_ticklabels([str(int(s)) for s in ticks], fontsize=21)
        else:

            ticks = [1000, 2500, 10000,25000, 100000, 200000]
            clb = fig.colorbar(sc)
            clb.set_ticks(ticks)
            clb.set_ticklabels([str(int(s)) for s in ticks], fontsize=21)

        for simplex in self.hull.simplices:
            plt.plot(
                self.points[simplex, 0]/max_x, self.points[simplex, 1]/max_y, "o-", color="black"
            )

        if hline is not None:
            plt.axhline(y=hline/max_y, color="red", linestyle="--", linewidth=2, label="Gap Cnstr.")
        if vline is not None:
            plt.axvline(x=vline/max_x, color="navy", linestyle="--", label="Best Ref.")


        ax1.axes.xaxis.set_visible(True)
        ax1.axes.yaxis.set_visible(True)

        if self.quantity == "dipole":
            plt.legend(loc="upper right", fontsize=21)
        else:
            plt.legend(loc="upper left", fontsize=21)

        ax1.grid(True, linestyle='--', linewidth=0.5, color='grey')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #pdb.set_trace()
        if coloring == "encounter":
            plt.savefig("{}_pareto_encounter.png".format(name_plot), dpi=600)
        if coloring == "density":
            #pdb.set_trace()
            
            
            plt.savefig("{}_pareto_count.png".format(name_plot), dpi=600)

        plt.close("all")

    def plot_trajectory(self, TRAJECTORY):
        """
        Plot the trajectory in propery space.
        """

        fs = 24

        plt.rc("font", size=fs)
        plt.rc("axes", titlesize=fs)
        plt.rc("axes", labelsize=fs)
        plt.rc("xtick", labelsize=fs)
        plt.rc("ytick", labelsize=fs)
        plt.rc("legend", fontsize=fs)
        plt.rc("figure", titlesize=fs)

        fig, ax1 = plt.subplots(figsize=(8, 8))
        p1 = TRAJECTORY["X_QUANTITY"].values
        p2 = TRAJECTORY["HOMO_LUMO_gap"].values
        step = np.arange(len(p1))
        sc = ax1.scatter(p1, p2, s=4, c=step)
        plt.xlabel(self.quantity + " (a.u.)", fontsize=21)
        plt.ylabel(
            "Gap" + " (a.u.)",
            fontsize=21,
            rotation=0,
            ha="left",
            y=1.05,
            labelpad=-50,
            weight=500,
        )
        clb = plt.colorbar(sc)
        clb.set_label("step")

        ax1 = make_pretty(ax1)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("steps.png", dpi=600)
        plt.close("all")

    def plot_result_spread(self, HISTOGRAMS, label):
        """
        Analyze the spread of the results accross different seeds.
        """

        plt.close("all")
        fs = 24

        plt.rc("font", size=fs)
        plt.rc("axes", titlesize=fs)
        plt.rc("axes", labelsize=fs)
        plt.rc("xtick", labelsize=fs)
        plt.rc("ytick", labelsize=fs)
        plt.rc("legend", fontsize=fs)
        plt.rc("figure", titlesize=fs)
        fig2, ax2 = plt.subplots(figsize=(8, 8))

        DIFFERENT_SEEDS = pd.DataFrame()
        for ind, hist in enumerate(HISTOGRAMS):
            Front = self.pareto(hist)
            BEST_SEED_ind = Front.sort_values(label, ascending=True).head(1)
            print(ind, BEST_SEED_ind)
            DIFFERENT_SEEDS = DIFFERENT_SEEDS.append(BEST_SEED_ind)

        sns.rugplot(data=DIFFERENT_SEEDS, x=label, palette="crest", height=0.1, ax=ax2)
        sns.violinplot(x=DIFFERENT_SEEDS[label], palette="Set3", ax=ax2)
        ax2.set_ylabel("#")
        ax2  = make_pretty(ax2)
        plt.tight_layout()

        plt.savefig("spread.png", dpi=300)

    def plot_chem_space(self, HISTOGRAM, label="loss"):
        """
        Make a PCA plot of the chemical space.
        """

        try:
            import seaborn as sns

            sns.set_context("poster")
            sns.set_style("whitegrid")
        except:
            pass

        fs = 24

        plt.rc("font", size=fs)
        plt.rc("axes", titlesize=fs)
        plt.rc("axes", labelsize=fs)
        plt.rc("xtick", labelsize=fs)
        plt.rc("ytick", labelsize=fs)
        plt.rc("legend", fontsize=fs)
        plt.rc("figure", titlesize=fs)
        print("Plot Chemical Space PCA")
        fig2, ax2 = plt.subplots(figsize=(8, 8))

        ax2 = make_pretty(ax2)

        MOLS = HISTOGRAM["SMILES"]
        P = HISTOGRAM[label].values

        if self.verbose:
            print("Compute PCA")
        X_2d = self.compute_PCA(MOLS)

        if self.verbose:
            print("Plot PCA")

        sc = ax2.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            s=50,
            alpha=0.1,
            marker="o",
            c=P,
            edgecolors="none",
        )

        ax2.set_xlabel("PC1", fontsize=21)
        ax2.set_ylabel(
            "PC2", fontsize=21, rotation=0, ha="left", y=1.05, labelpad=-50, weight=500
        )

        clb = plt.colorbar(sc)
        clb.set_label("Loss")

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.savefig("PCA.pdf")
        plt.savefig("PCA.png", dpi=600)
        plt.close("all")



class Chem_Div:
    def __init__(self, traj, subsample=1, verbose=False):
        self.traj = traj
        self.N    = np.arange(len(self.traj))
        self.subsample = subsample
        self.verbose = verbose

        self.traj = self.traj[::subsample]
        self.N   = self.N[::subsample]


    def compute_representations(self):
        """
        Compute the representations of all unique rdkit mols in the random walk.
        """
        self.rdkit_mols =  [Chem.MolFromSmiles(smi) for smi in  self.traj]
        self.X = rdkit_descriptors.get_all_FP(self.rdkit_mols, nBits=2048, fp_type="MorganFingerprint")


    def compute_diversity_i(self,i):
        """
        Compute PCA
        """
        #https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
        svd = TruncatedSVD() #n_components=2)
        svd.fit(self.X[:i])
        sing_vals = svd.singular_values_
        return np.linalg.norm(sing_vals)

    def compute_diversity(self):
        """
        Compute the diversity of a trajectory.
        """
        if self.verbose:
            print("Compute Diversity")
        self.diversity = []
        self.N = range(50, len(self.traj))
        for i in tqdm(self.N, disable=not self.verbose):
            self.diversity.append(self.compute_diversity_i(i) )




class Analyze_Chemspace:
    def __init__(self, path, full_traj=False, verbose=False):

        """
        mode : either optimization of dipole and gap = "optimization" or
               sampling locally in chemical space = "sampling"
        """

        self.path = path
        self.results = glob.glob(path)
        self.verbose = verbose
        self.full_traj = full_traj
        print(self.results)

    def parse_results(self):
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS = []
        ALL_TRAJECTORIES = []

        for run in tqdm(self.results, disable=not self.verbose):

            restart_data = loadpkl(run, compress=True)

            HISTOGRAM = self.to_dataframe(restart_data["histogram"])
            HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
            ALL_HISTOGRAMS.append(HISTOGRAM)
            if self.full_traj:
                traj = np.array(ordered_trajectory_from_restart(restart_data))
                CURR_TRAJECTORIES = []
                for T in range(traj.shape[1]):
                    sel_temp = traj[:, T]
                    TRAJECTORY = self.to_dataframe(sel_temp)
                    CURR_TRAJECTORIES.append(TRAJECTORY)
                ALL_TRAJECTORIES.append(CURR_TRAJECTORIES)

        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=["SMILES"])

        return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES



    def convert_from_tps(self, mols):
        """
        Convert the list of trajectory points molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """

        SMILES = []
        VALUES = []


        for tp in mols:
            try:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["canonical_rdkit"][-1])
                VALUES.append(curr_data["chemspacesampler"])
            except:
                if self.verbose:
                    print("Could not convert to smiles")
                pass

        SMILES, VALUES = self.make_canon(SMILES), np.array(VALUES)

        return SMILES, VALUES


    def compute_representations(self, MOLS, nBits):
        """
        Compute the representations of all unique smiles in the random walk.
        """

        X = rdkit_descriptors.get_all_FP(MOLS, nBits=nBits, fp_type="MorganFingerprint")
        return X

    def compute_PCA(self, MOLS, nBits=2048):
        """
        Compute PCA
        """

        X = self.compute_representations(MOLS, nBits=nBits)
        reducer = PCA(n_components=2)
        reducer.fit(X)
        X_2d = reducer.transform(X)
        return X_2d

    def to_dataframe(self, obj):
        """
        Convert the trajectory point object to a dataframe
        and extract xTB values if available.
        """

        df = pd.DataFrame()

        SMILES, VALUES = self.convert_from_tps(obj)
        df["SMILES"] = SMILES
        df["VALUES"] = VALUES
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def count_shell_value(self, curr_h, epsilon, return_mols=False):
        in_interval = curr_h["VALUES"] == -epsilon

        if return_mols:
            return in_interval.sum(), curr_h["SMILES"][in_interval].values
        else:
            return in_interval.sum()

    def count_shell(
        self, X_init, SMILES_sampled, dl, dh, nBits=2048, return_mols=False
    ):
        """
        Count the number of molecules in
        the shell of radius dl and dh.
        """
        darr = np.zeros(len(SMILES_sampled))
        for i, s in enumerate(SMILES_sampled):
            darr[i] = np.linalg.norm(
                X_init - self.compute_representations([s], nBits=nBits)
            )

        in_interval = (darr >= dl) & (darr <= dh)
        N = len(darr[in_interval])

        if return_mols == False:
            return N
        else:
            return N, SMILES_sampled[in_interval][:1000]

    def make_canon(self, SMILES):
        """
        Convert to canonical smiles form.
        """

        CANON_SMILES = []
        for smi in SMILES:

            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            CANON_SMILES.append(can)

        return CANON_SMILES

    def volume_of_nsphere(self, N, d):
        """
        Compute the volume of a n-sphere of radius d.
        """

        import mpmath

        N = mpmath.mpmathify(N)
        d = mpmath.mpmathify(d)

        return (mpmath.pi ** (N / 2)) / (mpmath.gamma(N / 2 + 1)) * d**N


