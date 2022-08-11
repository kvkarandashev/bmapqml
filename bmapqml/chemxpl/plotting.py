
from bmapqml.utils import * 
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import trajectory_point_to_canonical_rdkit
from bmapqml.chemxpl.random_walk import ordered_trajectory
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from rdkit import RDLogger  
import glob
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import seaborn as sns

lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)   
import warnings
warnings.filterwarnings('ignore')


class Analyze:
    """
    Analysis of the results of the optimization.
    """

    def __init__(self, path, verbose=False):


        self.path = path
        #pdb.set_trace()
        self.results = glob.glob(path)
        self.verbose = verbose

    def parse_results(self):
        
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS   = []
        ALL_TRAJECTORIES = []
        
        for run in tqdm(self.results, disable= not self.verbose):

            obj = loadtar(run)
            
            HISTOGRAM = self.to_dataframe(obj["histogram"])
            ALL_HISTOGRAMS.append(HISTOGRAM)
            traj = np.array(ordered_trajectory(obj["histogram"]))
            CURR_TRAJECTORIES = []
            for T in range(traj.shape[1]):
                sel_temp = traj[:,T]
                TRAJECTORY = self.to_dataframe(sel_temp)
                CURR_TRAJECTORIES.append(TRAJECTORY)
            ALL_TRAJECTORIES.append(CURR_TRAJECTORIES)

        del obj, traj

        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=['SMILES'])
        self.LABELS = self.GLOBAL_HISTOGRAM.columns[1:]

        if len(self.LABELS) > 0:
            if self.verbose:
                print("Best 5 molecules")

            for ind, label in enumerate(self.LABELS):
                print("{}".format(label))
                if ind <2:
                    BEST = self.GLOBAL_HISTOGRAM.sort_values(label, ascending=True).tail()[::-1]
                else:
                    BEST = self.GLOBAL_HISTOGRAM.sort_values(label, ascending=False).tail()[::-1]

                print("==========================================================")
                print(BEST)
                print("==========================================================")
                    
            return self.ALL_HISTOGRAMS,self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES
        
        else:
            if self.verbose:
                print("No Values could be extracted, only the SMILES were saved")
            return self.ALL_HISTOGRAMS,self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES


    def export_csv(self, HISTOGRAM):
        """
        Export the histogram to a csv file.
        """

        HISTOGRAM.to_csv("results.csv", index=False)

    def pareto(self,HISTOGRAM, maxX = True, maxY = True):
        
        """
        Filter the histogram to keep only the pareto optimal solutions.
        """

        Xs, Ys = HISTOGRAM["Dipole"].values, HISTOGRAM["HOMO_LUMO_gap"].values
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
                if p1==v1 and p2==v2:
                    inds.append(ind)
                ind+=1


        PARETO =  HISTOGRAM.iloc[np.array(inds)]
        if self.verbose:
            print("Pareto optimal solutions:")
            print(PARETO.sort_values("xTB_MMFF_min_en_conf_electrolyte"))

        return PARETO


    def plot_trajectory_loss(self, ALL_TRAJECTORIES):
        """
        Compute the loss of the trajectories.
        """

        fs = 24

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 

        fig,ax1= plt.subplots(figsize=(8,8))

        for traj in ALL_TRAJECTORIES:
            n_temps = len(traj)
            cmap   = cm.coolwarm(np.linspace(0, 1, n_temps))
            for c, T in zip(cmap,range(n_temps)[:3]):

                sel_temp = traj[T]["xTB_MMFF_min_en_conf_electrolyte"]
                N = np.arange(len(sel_temp))
                ax1.scatter(N,sel_temp,s=5, color=c, alpha=0.5)
                ax1.plot(N,sel_temp,"-", alpha=0.1)
    
        ax1.spines['right'].set_color('none')
        ax1.spines['top'].set_color('none')
        ax1.spines['bottom'].set_position(('axes', -0.05))
        ax1.spines['bottom'].set_color('black')
        ax1.spines['left'].set_color('black')
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.spines['left'].set_position(('axes', -0.05))

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("loss.pdf")  
        plt.savefig("loss.png")
        plt.close("all")


    def plot_pareto(self, HISTOGRAM, PARETO, ALL_PARETOS=False):

        fs = 24

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 

        fig,ax1= plt.subplots(figsize=(8,8))
        P1  = HISTOGRAM["Dipole"].values
        P2  = HISTOGRAM["HOMO_LUMO_gap"].values
        summe = HISTOGRAM["xTB_MMFF_min_en_conf_electrolyte"].values

        sc = ax1.scatter(P1, P2,s = 4, c=summe)
        plt.xlabel("Dipole"  + " (a.u.)", fontsize=21)
        plt.ylabel("Gap" + " (a.u.)", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
        clb = plt.colorbar(sc)
        clb.set_label("Loss") 

        ax1.spines['right'].set_color('none')
        ax1.spines['top'].set_color('none')
        ax1.spines['bottom'].set_position(('axes', -0.05))
        ax1.spines['bottom'].set_color('black')
        ax1.spines['left'].set_color('black')
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.spines['left'].set_position(('axes', -0.05))

        plt.plot(PARETO["Dipole"],PARETO["HOMO_LUMO_gap"],'o', color='black')
        plt.plot(PARETO["Dipole"],PARETO["HOMO_LUMO_gap"],'k-',linewidth=2)

        alpha = 0.5
        if ALL_PARETOS!=False:
            cmap   = iter(cm.rainbow(np.linspace(0, 1, len(ALL_PARETOS))))
            for color, hist in zip(cmap, ALL_PARETOS):
                SUB_PARETO = self.pareto(hist)
                plt.plot(SUB_PARETO["Dipole"],SUB_PARETO["HOMO_LUMO_gap"],'o',alpha=alpha, color=color)
                plt.plot(SUB_PARETO["Dipole"],SUB_PARETO["HOMO_LUMO_gap"],'-',color=color, alpha=alpha ,linewidth=0.5)


        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("pareto.pdf")  
        plt.savefig("pareto.png")
        plt.close("all")      



    def plot_trajectory(self, TRAJECTORY):
            
            fs = 24
    
            plt.rc('font', size=fs)
            plt.rc('axes', titlesize=fs)
            plt.rc('axes', labelsize=fs)           
            plt.rc('xtick', labelsize=fs)          
            plt.rc('ytick', labelsize=fs)          
            plt.rc('legend', fontsize=fs)   
            plt.rc('figure', titlesize=fs) 
    
            fig,ax1= plt.subplots(figsize=(8,8))
            p1   = TRAJECTORY["Dipole"].values
            p2   = TRAJECTORY["HOMO_LUMO_gap"].values
            step = np.arange(len(p1))
            sc = ax1.scatter(p1, p2,s =4, c=step)
            plt.xlabel("Dipole"  + " (a.u.)", fontsize=21)
            plt.ylabel("Gap" + " (a.u.)", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
            clb = plt.colorbar(sc)
            clb.set_label("step")     

            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')
            ax1.spines['bottom'].set_position(('axes', -0.05))
            ax1.spines['bottom'].set_color('black')
            ax1.spines['left'].set_color('black')
            ax1.yaxis.set_ticks_position('left')
            ax1.xaxis.set_ticks_position('bottom')
            ax1.spines['left'].set_position(('axes', -0.05))

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("steps.pdf")  
            plt.savefig("steps.png")  
            plt.close("all")         



    def plot_result_spread(self, HISTOGRAMS, label):
        
        """
        Analyze the spread of the results accross different seeds.
        """

        plt.close("all")
        fs = 24

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 
        fig2,ax2= plt.subplots(figsize=(8,8))

        DIFFERENT_SEEDS = pd.DataFrame()
        for ind, hist in enumerate(HISTOGRAMS):
            Front = self.pareto(hist)
            BEST_SEED_ind = Front.sort_values(label, ascending=True).head(1)
            print(ind, BEST_SEED_ind)
            DIFFERENT_SEEDS = DIFFERENT_SEEDS.append(BEST_SEED_ind)

        sns.rugplot(data=DIFFERENT_SEEDS, x=label,palette="crest", height=.1, ax=ax2)
        sns.violinplot(x=DIFFERENT_SEEDS[label], palette="Set3", ax=ax2)
        ax2.set_ylabel("#")
        ax2.spines['right'].set_color('none')
        ax2.spines['top'].set_color('none')
        ax2.spines['bottom'].set_position(('axes', -0.05))
        ax2.spines['bottom'].set_color('black')
        ax2.spines['left'].set_color('black')
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['left'].set_position(('axes', -0.05))
        plt.tight_layout()

        plt.savefig("spread.pdf")
        plt.savefig("spread.png")

    def compute_representations(self, MOLS, nBits):
        """
        Compute the representations of all unique smiles in the random walk.
        """

        X = rdkit_descriptors.get_all_FP(MOLS,nBits=nBits, fp_type="MorganFingerprint")
        return X

    def compute_PCA(self, MOLS, nBits=4096):
        """
        Compute PCA
        """

        X = self.compute_representations(MOLS, nBits=nBits)
        reducer = PCA(n_components=2)
        reducer.fit(X)
        X_2d = reducer.transform(X)
        return X_2d




    def plot_chem_space(self, HISTOGRAM,  label="xTB_MMFF_min_en_conf_electrolyte"):
        """
        Make a PCA plot of the chemical space.
        """

        try:
            import seaborn as sns
            sns.set_context("poster")
            sns.set_style('whitegrid')
        except:
            pass

        fs = 24

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 
        print("Plot Chemical Space PCA")
        fig2,ax2= plt.subplots(figsize=(8,8))

        ax2.spines['right'].set_color('none')
        ax2.spines['top'].set_color('none')
        ax2.spines['bottom'].set_position(('axes', -0.05))
        ax2.spines['bottom'].set_color('black')
        ax2.spines['left'].set_color('black')
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['left'].set_position(('axes', -0.05))


        MOLS = HISTOGRAM["SMILES"]
        P    = HISTOGRAM[label].values

        if self.verbose:
            print("Compute PCA")
        X_2d = self.compute_PCA(MOLS)

        if self.verbose:
            print("Plot PCA")


        sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=50,alpha=0.1,marker="o", c=P,edgecolors='none')

        ax2.set_xlabel("PC1", fontsize=21)
        ax2.set_ylabel("PC2", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)

        clb = plt.colorbar(sc)
        clb.set_label('Loss')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("PCA.pdf")
        plt.savefig("PCA.png")
        plt.close("all")



    def to_dataframe(self, obj):
        """
        Convert the object to a dataframe.
        """
        df = pd.DataFrame()
        SMILES = self.convert_to_smiles(obj)
        df["SMILES"] = SMILES

        try:
            values, labels = self.extract_values(obj)
            for l in labels:
                df[l] = values[:,labels.index(l)]

        except:
            if self.verbose:
                print("Error in reading values")
        
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df


    def convert_to_smiles(self, mols):
        """
        Convert the list of molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """

        smiles_mol = []
        for tp in mols:
            rdkit_obj = trajectory_point_to_canonical_rdkit(tp)
            smiles_mol.append(rdkit_obj[3])

        smiles_mol = self.make_canon(smiles_mol)
        return smiles_mol

    def extract_values(self, mols):
        """
        Extract the values from the trajectory points.
        tp_list: list of molecules as trajectory points
        """

        labels = [l for l in mols[0].calculated_data.keys()]
        values = []
        for tp in mols:
            values.append(list(tp.calculated_data.values()))
        
        values = np.float_(np.array(values))

        return values, labels

    def make_canon(self,SMILES):
        
        """
        Convert to canonical smiles form.
        """
    
        CANON_SMILES = []
        for smi in SMILES:
            
            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            CANON_SMILES.append(can)


        return CANON_SMILES


    def count_shell(self, X_init, X_sampled, dl, dh, nBits=4096):

        """
        Count the number of molecules in 
        the shell of radius dl and dh.
        """

        darr = np.zeros(len(X_sampled))
        for i, xs in  enumerate(X_sampled):
            darr[i] = np.linalg.norm(X_init - xs)


        in_interval = ((darr >= dl) & (darr <=  dh))
        N = len(darr[in_interval])

        try:
            import mpmath
            dV = self.volume_of_nsphere(nBits, dh) - self.volume_of_nsphere(nBits, dl)
            logRDF =   float(mpmath.log(N/dV))
            return N, logRDF

        except ImportError:
            print("mpmath not installed")
            return N, None


    def volume_of_nsphere(self,N, d):
        import mpmath
        
        N = mpmath.mpmathify(N)
        d = mpmath.mpmathify(d)

        return (mpmath.pi ** (N / 2)) / (mpmath.gamma(N / 2 + 1)) * d ** N