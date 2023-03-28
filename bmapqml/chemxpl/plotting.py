from bmapqml.utils import *
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import trajectory_point_to_canonical_rdkit
from bmapqml.chemxpl.random_walk import ordered_trajectory_from_restart
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from rdkit import RDLogger
import glob
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import seaborn as sns
import random
import joblib
import time
from matplotlib.ticker import FormatStrFormatter
import pdb

np.random.seed(122)
random.seed(122)
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
            print("Loading data from {}".format(run))
            start = time.time()
            restart_data = loadpkl(run, compress=False)
            done = time.time()
            elapsed = done - start
            print(elapsed)

            print("Data loaded from {}".format(run))
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


            self.ALL_TRAJECTORIES = pd.concat(ALL_TRAJECTORIES[0])
            self.X_QUANTITY_traj, self.GAP_traj =  self.ALL_TRAJECTORIES["X_QUANTITY"].values, self.ALL_TRAJECTORIES["HOMO_LUMO_gap"]
            return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES
        else:
            return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM


    def pareto(self, mode="trajectory"):
        try:
            from scipy.spatial import ConvexHull
        except:
            print("Please install scipy")
            exit()

        if mode == "trajectory":
            self.points = np.array([self.X_QUANTITY_traj, self.GAP_traj]).T
            self.hull = ConvexHull(self.points)
            pareto = np.unique(self.hull.simplices.flatten())
            self.PARETO = self.ALL_TRAJECTORIES.iloc[pareto]
            self.PARETO = self.PARETO.sort_values(by=['X_QUANTITY'], ascending=False)
            return self.PARETO


        else:
            self.points = np.array([self.X_QUANTITY, self.GAP]).T
            self.hull = ConvexHull(self.points)
            pareto = np.unique(self.hull.simplices.flatten())
            self.PARETO = self.GLOBAL_HISTOGRAM.iloc[pareto]
            self.PARETO = self.PARETO.sort_values(by=['X_QUANTITY'], ascending=False)

            
            return self.PARETO


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

        VALUES = np.array(VALUES)
        #SMILES, VALUES = self.make_canon(SMILES), np.array(VALUES)

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

    def plot_pareto(self,name_plot, hline=None, vline=None,dataset="QM9", coloring="encounter", plot_quantity = "solvation_energy",labels=False):
        """
        Plot the pareto optimal solutions.
        """

        fig, ax1 = plt.subplots(figsize=(8, 8))

        cc = 'gnuplot_r'
        gs = 30

        max_x = abs(vline)
        max_y = 0.6193166670127856
        
        
        if hline is not None:
            plt.axhline(y=hline/max_y, color="red", linestyle="--", linewidth=2, label="Gap Cnstr.")
        if vline is not None:
            plt.axvline(x=vline/max_x, color="navy", linestyle="--", label="Best Ref.")


        ymin_tick = hline/max_y - 0.05 #0 if hline is None else hline
        if coloring == "encounter":
            sc=ax1.hexbin(self.X_QUANTITY/max_x,  self.GAP/max_y, gridsize=gs, mincnt=5, cmap=cc,C=self.ENCOUNTER, linewidths=0.2)
        if coloring == "density":
            from matplotlib import colors
            np.seterr(under='warn')
            sc=ax1.hexbin(self.X_QUANTITY_traj/max_x, self.GAP_traj /max_y, gridsize=gs,bins="log", mincnt=5,linewidths=0.2,norm=colors.LogNorm(vmin=100, vmax=200000))

            
        if labels:
            if plot_quantity == "solvation_energy":
                if dataset == "QM9":
                    #ax1.set_xlabel(r"$\Delta G_{\mathrm{solv.}}/ \mathrm{max}(  \vert \Delta G_{\mathrm{solv.}}^{\mathrm{QM9}} \vert)$", fontsize=fs)
                    #ax1.set_xlim(-1.5,0.1)
                    ax1.set_ylim(ymin_tick,1.05)
                    #ticks = np.arange(-1.5,0.5,0.2)
                    #ax1.set_xticks(xticks)
                    #ax1.set_xticklabels([str(round(x,1)) for x in xticks])
                    #ax1.set_xticklabels([str(round(x,1)) for x in xticks])

                if dataset == "EGP":
                    #ax1.set_xlabel(r"$\Delta G_{\mathrm{solv.}}/ \mathrm{max}(  \vert \Delta G_{\mathrm{solv.}}^{\mathrm{EGP}} \vert)$", fontsize=fs)
                    #ax1.set_xlim(-14,2)
                    ax1.set_ylim(ymin_tick,1.05)
                    #xticks = np.arange(-12,2,2)
                    ##ax1.set_xticks(xticks)
                    #ax1.set_xticklabels([str(round(x,1)) for x in xticks])

            if plot_quantity == "dipole":
                if dataset == "QM9":
                    #ax1.set_xlabel(r"$ D / \mathrm{max}(  D^{\mathrm{QM9}} )$", fontsize=fs)
                    pass
                    ax1.set_ylim(ymin_tick,1.05)
                    #ax1.set_xlim(0.0,1.5)
                    #xticks  = np.arange(0.0,2.0,0.5)
                    #ax1.set_xticks(xticks)
                    #only one number after comma
                    #ax1.set_xticklabels([str(round(x,1)) for x in xticks])
                if dataset == "EGP":
                    #ax1.set_xlabel(r"$ D / \mathrm{max}(  D^{\mathrm{EGP}} )$", fontsize=fs)
                    #ax1.set_xlim(0,9.0)
                    ax1.set_ylim(ymin_tick,1.05)
                    #xticks  = np.arange(0,10,2)
                    #ax1.set_xticks(xticks)
                    #ax1.set_xticklabels([str(x) for x in xticks])
                    pass

            if dataset == "QM9":
                pass
                #plt.ylabel(
                #    r"$\Delta \epsilon / \mathrm{max}(  \Delta \epsilon^{\mathrm{QM9}} )$",
                #    fontsize=fs,
                #    rotation=0,
                #    ha="left",
                #    y=1.05,
                #    labelpad=-50,
                #    weight=500,
                #)
            if dataset == "EGP":
                pass
                #plt.ylabel(
                #    r"$\Delta \epsilon / \mathrm{max}(  \Delta \epsilon^{\mathrm{EGP}} )$",
                #    fontsize=fs,
                #    rotation=0,
                #    ha="left",
                #    y=1.05,
                #    labelpad=-50,
                #    weight=500,
                #)
        
        
        
        ax1 = make_pretty(ax1)
        ax1.xaxis.set_tick_params(labelsize=30)
        ax1.yaxis.set_tick_params(labelsize=30)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if labels:
            if coloring == "encounter":
                clb = fig.colorbar(sc)
                clb.ax.set_title('#',fontsize=fs)
                ticks =  [i*5000 for i in range(1,7)]
                clb.set_ticks(ticks)
                clb.set_ticklabels([str(int(s)) for s in ticks], fontsize=fs)
            else:

                ticks = [1000, 2500, 10000,25000, 100000, 200000]
                #clb = fig.colorbar(sc)
                #clb.ax.set_title('#',fontsize=fs)
                #clb.set_ticks(ticks)
                #clb.set_ticklabels([str(int(s)) for s in ticks], fontsize=fs)

        for simplex in self.hull.simplices:
            plt.plot(
                self.points[simplex, 0]/max_x, self.points[simplex, 1]/max_y, "o-", color="black"
            )



        ax1.axes.xaxis.set_visible(True)
        ax1.axes.yaxis.set_visible(True)

        if labels:
            if self.quantity == "dipole":
                #plt.legend(loc="upper right", fontsize=fs)
                pass
            else:
                pass
                #plt.legend(loc="upper left", fontsize=fs)

        ax1.grid(True, linestyle='--', linewidth=0.5, color='grey')
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if coloring == "encounter":
            plt.savefig("{}_enc.png".format(name_plot), dpi=600)
            #plt.savefig("{}_enc.svg".format(name_plot))

        if coloring == "density":    
            plt.savefig("{}_den.png".format(name_plot), dpi=600)
            #plt.savefig("{}_den.svg".format(name_plot))

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


        # deactivate colorbar
        #clb = plt.colorbar(sc)
        #clb.set_label("step")

        ax1 = make_pretty(ax1)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("steps.png", dpi=600)
        plt.close("all")

    def plot_result_spread(self, HISTOGRAMS, label):
        """
        Analyze the spread of the results accross different seeds.
        """

        plt.close("all")
        

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

        ax2.set_xlabel("PC1", fontsize=fs)
        ax2.set_ylabel(
            "PC2", fontsize=fs, rotation=0, ha="left", y=1.05, labelpad=-50, weight=500
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
        svd = TruncatedSVD()
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
    def __init__(self, path,rep_type="2d" ,full_traj=False, verbose=False):

        """
        mode : either optimization of dipole and gap = "optimization" or
               sampling locally in chemical space = "sampling"
        """

        self.path = path
        self.rep_type = rep_type
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


        if self.full_traj:
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
        else:
            ALL_HISTOGRAMS = joblib.Parallel(n_jobs=8)(joblib.delayed(self.process_run)(run) for run in tqdm(self.results))


        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=["SMILES"])

        return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES


    def process_run(self, run):
        restart_data = loadpkl(run, compress=True)

        HISTOGRAM = self.to_dataframe(restart_data["histogram"])
        HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
        return HISTOGRAM




    def convert_from_tps(self, mols):
        """
        Convert the list of trajectory points molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """


        if self.rep_type == "2d":
            SMILES = []
            VALUES = []

            
            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["canonical_rdkit"][-1])
                VALUES.append(curr_data["chemspacesampler"])


            VALUES = np.array(VALUES)

            return SMILES, VALUES
        if self.rep_type == "3d":
            """
            (Pdb) tp.calculated_data["morfeus"].keys()
            dict_keys(['coordinates', 'nuclear_charges', 'canon_rdkit_SMILES', 'rdkit_energy', 'rdkit_degeneracy', 'rdkit_Boltzmann'])
            """



            SMILES = []
            VALUES = []
            #len(tp.calculated_data["morfeus"]["coordinates"])
            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["morfeus"]["canon_rdkit_SMILES"])
                VALUES.append(curr_data["3d"])

            VALUES = np.array(VALUES)

            return SMILES, VALUES


    def compute_representations(self, MOLS, nBits):
        """
        Compute the representations of all unique smiles in the random walk.
        """

        X = rdkit_descriptors.get_all_FP(MOLS, nBits=nBits, fp_type="MorganFingerprint")
        return X

    def compute_projection(self, MOLS, nBits=2048, clustering=False, projector="PCA"):
        


        X = self.compute_representations(MOLS, nBits=nBits)
        if projector == "UMAP":
            import umap
            reducer = umap.UMAP(random_state=42)
        if projector == "PCA":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)

        reducer.fit(X)
        X_2d = reducer.transform(X)

        if clustering == False:
            return X_2d


        else:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            sil_scores = []
            for n_clusters in range(2, 6):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(X_2d)
                labels = kmeans.labels_
                sil_scores.append(silhouette_score(X_2d, labels))

            optimal_n_clusters = np.argmax(sil_scores) + 2 #5
            print(optimal_n_clusters)
            kmeans = KMeans(n_clusters=optimal_n_clusters,tol=1e-8,max_iter=500,random_state=0)
            kmeans.fit(X_2d)
            labels = kmeans.labels_
            
            # Assign each molecule to the closest cluster
            clusters = [[] for _ in range(optimal_n_clusters)]
            cluster_X_2d = [[] for _ in range(optimal_n_clusters)]
            for i, label in enumerate(labels):
                clusters[label].append(MOLS[i])
                cluster_X_2d[label].append(X_2d[i])
            
            # Find most representative molecule for each cluster
            representatives = []
            indices = []

            #kmeans.cluster_centers_ is possibly wrong
            X_rep_2d = []
            for label, cluster in enumerate(clusters):
                
                distances = [np.linalg.norm(x - kmeans.cluster_centers_[label]) for x in cluster_X_2d[label]]
                #
                representative_index = np.argmin(distances)
                representatives.append(cluster[representative_index])
                indices.append(representative_index)
                X_rep_2d.append(cluster_X_2d[label][representative_index])

            
            #pdb.set_trace()
            print(kmeans.cluster_centers_)
            #pdb.set_trace()
            SMILES_rep = [Chem.MolToSmiles(mol) for mol in representatives]
            X_rep_2d = np.array(X_rep_2d)
            return X_2d ,clusters,cluster_X_2d, X_rep_2d,SMILES_rep,reducer









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


