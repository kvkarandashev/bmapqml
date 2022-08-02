
from bmapqml.utils import * 
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import trajectory_point_to_canonical_rdkit
from bmapqml.chemxpl.random_walk import ordered_trajectory
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
import numpy as np
from rdkit import RDLogger  
import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import seaborn as sns

#/bmapqml/chemxpl/utils.py
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
        self.results = os.listdir(path)[:1]
        self.verbose = verbose

    def parse_results(self):
        
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS   = []
        ALL_TRAJECTORIES = []

        for run in tqdm(self.results):

            obj = pickle.load(open(self.path+run, "rb"))
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

        GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = GLOBAL_HISTOGRAM

        # GLOBAL_HISTOGRAM.drop_duplicates(subset=['SMILES'])
        """
        Somehow be careful!!!
        (Pdb) GLOBAL_HISTOGRAM[GLOBAL_HISTOGRAM["SMILES"]=="OC(O)CC(F)C(O)F"]
                    SMILES    Dipole  HOMO_LUMO_gap  xTB_MMFF_electrolyte
        5591  OC(O)CC(F)C(O)F  1.330659       0.415468             -7.507930
        5442  OC(O)CC(F)C(O)F  3.610319       0.394474            -10.972133        
        """
        
        
        self.LABELS = GLOBAL_HISTOGRAM.columns[1:]

        if self.verbose:
            print("Best 5 molecules")

        for ind, label in enumerate(self.LABELS):
            print("{}".format(label))
            if ind <2:
                BEST = self.GLOBAL_HISTOGRAM.sort_values(label, ascending=True).tail()[::-1]
                print
            else:
                BEST = self.GLOBAL_HISTOGRAM.sort_values(label, ascending=False).tail()[::-1]

            print("==========================================================")
            print(BEST)
            print("==========================================================")
                
        return self.ALL_HISTOGRAMS,self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES


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
            print(PARETO.sort_values("xTB_MMFF_electrolyte"))

        return  PARETO

    def pareto_plot(self, HISTOGRAM, PARETO, ALL_PARETOS=False):


        fs = 24

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 

        fig,ax1= plt.subplots(figsize=(8,8))
        p1  = HISTOGRAM["Dipole"].values
        p2  = HISTOGRAM["HOMO_LUMO_gap"].values
        summe = HISTOGRAM["xTB_MMFF_electrolyte"].values
        xi = np.linspace(min(p1), max(p1), 1000)
        yi = np.linspace(min(p2), max(p2), 1000)
        #pdb.set_trace()
        triang = tri.Triangulation(p1, p2)
        interpolator = tri.LinearTriInterpolator(triang,summe)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        ax1.contour(xi, yi, zi, levels=18, linewidths=0.5, colors='k')
        sc = ax1.scatter(p1, p2,s =4, c=summe)
        plt.xlabel("Dipole"  + " [unit 1]", fontsize=21)
        plt.ylabel("HOMO_LUMO_gap" + " [unit 2]", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
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
        plt.close("all")      



    def trajectory_plot(self, TRAJECTORY):
            
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
            plt.xlabel("Dipole"  + " [unit 1]", fontsize=21)
            plt.ylabel("HOMO_LUMO_gap" + " [unit 2]", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
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
            plt.close("all")         



    def result_spread(self, HISTOGRAMS, label):
        
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

        #sns.kdeplot(data=DIFFERENT_SEEDS, x=label,fill=True, palette="crest",cut=0, ax=ax2)
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


    def to_dataframe(self, obj):
        """
        Convert the object to a dataframe.
        """
        df = pd.DataFrame()
        values, labels = self.extract_values(obj)
        SMILES = self.convert_to_smiles(obj)
        df["SMILES"] = SMILES
        for l in labels:
            df[l] = values[:,labels.index(l)]
        
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




class analyze_random_walk:

    """
    Class for analyzing a random walk after the simulation.
    Visualize chemical space and convex hull of the walk.
    """

    def __init__(self, histograms,target, trajectory=None,fp_type=None, model=None, verbose=False, name=None, dmin=None, thickness=None):
        """
        histogram : list of all unique encountered points in the random walk.
        minimize_fct : if desired reevaluate all unique molecules using this function
        """

        self.name = name or None
        self.histogram = histograms or None
        self.target = target
        self.model = None or model
        self.trajectory = None or trajectory
        self.fp_type = fp_type
        self.dmin = dmin or None
        self.thickness = thickness or None
        print(self.fp_type)
        #pdb.set_trace()


        try:
            unique_tps = []
            for h in histograms:
                unique_tps.append(pickle.load(open(h, "rb")))
            self.tps     = np.concatenate(unique_tps)
            self.histogram= np.array(self.convert_to_smiles(self.tps))

        except:
            self.tps = pickle.load(open(histograms, "rb"))
            self.tps_smiles= np.array(self.convert_to_smiles(self.tps))
            self.histogram= np.array(self.convert_to_smiles(self.tps))

        if verbose:
            print("Number of unique points in the random walk:", len(self.histogram))
        
        if self.trajectory is not None:
            pass
            """
            print("Loading trajectory from file:", self.trajectory)
            self.all_tps = np.array(pickle.load(open(self.trajectory, "rb")))
            print("Done loading trajectory from file:", self.trajectory)
            self.all_tps_smiles = np.ones_like(self.all_tps)
            #here select a single temperature to make it easier
            self.all_tps = self.all_tps[:,2]

            self.all_tps_smiles = np.array(self.convert_to_smiles(self.all_tps))
            #[]
            #np.array(Parallel(n_jobs=12)(delayed(trajectory_point_to_canonical_rdkit)(tp) for tp in self.all_tps))[:,3]
            print("Done computing all smiles")
            
            """


        
    def convert_to_smiles(self, mols):
        
        """
        Convert the list of molecules to SMILES strings.
        tp_list: list of molecules as trajectory points
        smiles_mol: list of rdkit molecules
        """

        smiles_mol = []
        for tp in tqdm(mols):
            
            rdkit_obj = trajectory_point_to_canonical_rdkit(tp)
            smiles_mol.append(rdkit_obj[3])

        return smiles_mol


    def compute_representations(self):
        """
        Compute the representations of all unique smiles in the random walk.
        """

        X = rdkit_descriptors.get_all_FP(self.histogram, fp_type=self.fp_type)
        return X


    def evaluate_all_trajectory_points(self):
        """
        Evaluate the function on all trajectory points.
        """
        print("Evaluating all trajectory points")
        #X_target = rdkit_descriptors.extended_get_single_FP(self.target,self.fp_type )
        #pdb.set_trace()
        #X = rdkit_descriptors.get_all_FP(self.all_tps_smiles, fp_type=self.fp_type)
        print("evaluating all trajectory points")
        evaluation = self.model.evaluate_trajectory(self.all_tps)

        self.all_values = evaluation[:,0]
        self.all_distances = evaluation[:,1] #np.zeros(len(self.all_tps))

        #print("evaluating all distances")
        #for ind,x in enumerate(X):
        #    self.all_distances[ind] = np.linalg.norm(x-X_target)

        np.savez_compressed("complete_histogram_{}".format(self.name), all_tps_smiles=self.all_tps_smiles,all_distances=self.all_distances,all_values=self.all_values) 
        order = np.argsort(self.all_distances)
        print(self.all_tps_smiles,order)
        self.all_tps_smiles = self.all_tps_smiles[order]
        self.all_distances = self.all_distances[order]
        self.all_values = self.all_values[order]
        print("Done evaluating all trajectory points")
        trajectory_report = open("trajectory_report_{}.txt".format(self.name), "w")

        for smi, d, v in zip(self.all_tps_smiles, self.all_distances, self.all_values):
            trajectory_report.write("{}\t{}\t{}\n".format(smi, d, v))
            print(smi, d, v)
        trajectory_report.close()

        return self.all_tps_smiles, self.all_distances,self.all_values



    def evaluate_histogram(self):
        """
        Compute values of a function evaluated on all unique smiles in the random walk.
        """
        print("Evaluating histogram")
        #X_target = rdkit_descriptors.extended_get_single_FP(self.target,self.fp_type )

        evaluation = self.model.evaluate_trajectory(self.tps)
        self.values = evaluation[:,0]
        self.distances = evaluation[:,1]
        #np.zeros(len(self.tps))
        #X = rdkit_descriptors.get_all_FP(self.tps_smiles, fp_type=self.fp_type)
        #for ind,x in enumerate(X):
        #   self.distances[ind] = np.linalg.norm(x-X_target)

        lower_lim , upper_lim = self.dmin-self.thickness, self.dmin
        in_interval = ((self.distances >= lower_lim) & (self.distances <=  upper_lim))
        print("Number of points in the interval:", "[{},{}]".format(lower_lim, upper_lim)  , len(self.distances[in_interval]))
        print(self.tps_smiles[in_interval])


        #np.savez_compressed("histogram_{}".format(self.name), tps_smiles=self.tps_smiles,distances=self.distances,values=self.values) 
        #exit()
        #pdb.set_trace()
        
        return self.values


    def evaluate_RDF(self):
        import mpmath

        """
        Compute values of a function evaluated on all unique smiles in the random walk.
        """
        print("Evaluating RDF")

        evaluation = self.model.evaluate_trajectory(self.tps)
        self.values = evaluation[:,0]
        self.distances = evaluation[:,1]

        lower_lim , upper_lim = self.dmin-self.thickness, self.dmin 
        shell_vol = self.volume_of_nsphere(4096, upper_lim) - self.volume_of_nsphere(4096, lower_lim)
        
        in_interval = ((self.distances >= lower_lim) & (self.distances <=  upper_lim))
        N = len(self.distances[in_interval])
        logRDF =   mpmath.log(N/shell_vol)
        print("Number of points in the interval:", "[{},{}]".format(lower_lim, upper_lim)  , N )
        print(self.tps_smiles[in_interval])


        return lower_lim, upper_lim, N,logRDF




    def volume_of_nsphere(self,N, R):
        #from scipy.special import gamma
        import mpmath
        
        N = mpmath.mpmathify(N)
        R = mpmath.mpmathify(R)

        return (mpmath.pi ** (N / 2)) / (mpmath.gamma(N / 2 + 1)) * R ** N




    def compute_radial_distribution_functino(self, N, R):
        """
        Compute the radial distribution function 
        """
        pass

    def compute_PCA(self):
        """
        Compute PCA of the random walk and return the first two principal components.
        """
        X = self.compute_representations()
        reducer = PCA(n_components=2)
        reducer.fit(X)
        X_2d = reducer.transform(X)
        return X_2d, reducer

    def compute_pareto_front(self):
        """
        values: array of function values
        Returns values of points in the pareto front.
        pareto_front as well as the indices
        """

        if self.values.shape == (len(self.values),3):

            """
            Means two properties and the loss are considered in this run,
            self.values has shape (n,3) |property 1 and 2| and the loss
            Subselect only the two properties to determine pareto front
            """

            two_properties = self.values[:,:2]
            Xs, Ys = two_properties[:,0], two_properties[:,1]
            maxY = False
            sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
            pareto_front = [sorted_list[0]]
            
            for pair in sorted_list[1:]:
                if maxY:
                    if pair[1] >= pareto_front[-1][1]:
                        pareto_front.append(pair)
                else:
                    if pair[1] <= pareto_front[-1][1]:
                        pareto_front.append(pair)

            inds = []   
            for pair in pareto_front:
                if pair[0] in two_properties[:,0] and pair[1] in two_properties[:,1]:
                    inds.append( int(np.where(two_properties[:,0]==pair[0]) and np.where(two_properties[:,1]==pair[1])[0][0]))
            
            inds =np.array(inds)
            inds  = inds.astype('int')
            self.pareto_mols = self.histogram[inds]
            self.pareto_front = np.array(pareto_front)
            return self.pareto_front, np.int_(inds), self.pareto_mols


        if self.values.shape == (len(self.values),2):
            inds = np.argsort(self.values[:,0])
            self.pareto_mols = self.histogram[inds]
            self.pareto_front = np.array(self.values[inds])

            return self.pareto_front, np.int_(inds), self.pareto_mols


        if self.values.shape == (len(self.values),):

            """
            Only one property is considered in this run,
            self.values has shape (n,1) |property 1|
            """

            inds = np.argsort(self.values)
            self.pareto_mols = self.histogram[inds]
            self.pareto_front = np.array(self.values[inds])

            return self.pareto_front, np.int_(inds), self.pareto_mols

        else:
            print("Currently only one or two properties are supported.")
            raise ValueError("Wrong shape of values array")



    def write_report(self, filename, verbose=True):

        """
        Export molecules on the pareto front to a file.
        Include two properties that are also included in the 
        simulation
        """

        

        try:
            if self.values.shape == (len(self.values),3):
                front, _ , mols = self.compute_pareto_front()

                best_P1, best_P2, best_loss = np.argsort(front[:,0]),np.argsort(front[:,1]),np.argsort(self.values[:,2][:len(front)])
                if verbose:
                    print("Best Candidates:")
                    [print(m) for m in self.pareto_mols[best_loss]]
                    print("Most Stable:")
                    [print(m, "{:.2f}".format(P1)+" eV") for m,P1 in zip(mols[best_P1], front[:,0][best_P1])] 
                    print("Smallest Gap:")
                    [print(m,"{:.2f}".format(P2)+" eV") for m,P2 in zip(mols[best_P2], front[:,1][best_P2])]         

                f = open(filename+".dat", "w")
                f.write("Best Candidates:\n")
                [f.write(m+"\n") for m in self.pareto_mols[best_loss]]
                f.write("Most Stable:\n")
                [f.write(m+" "+"{:.2f}".format(P1)+" eV\n") for m,P1 in zip(mols[best_P1], front[:,0][best_P1])]
                f.write("Smallest Gap:\n")
                [f.write(m+" "+"{:.2f}".format(P2)+" eV\n") for m,P2 in zip(mols[best_P2], front[:,1][best_P2])]
                f.close()        


            elif self.values.shape == (len(self.values),2):
               
                front, _ , mols = self.compute_pareto_front()
                best_loss = np.argsort(front[:,0])
                if verbose:
                    print("Best Candidates:")
                    [print(m,  "{:.2f}".format(P1[1])) for m,P1 in zip(self.pareto_mols[best_loss], front[best_loss])]
                f = open(filename+".dat", "w")
                f.write("Best Candidates:\n")

            
                [f.write(m+" " + "{:.2f}".format(P1[1])+"\n") for m,P1 in zip(self.pareto_mols[best_loss], front[best_loss])]
                f.close()


            elif self.values.shape == (len(self.values),):

                front, _ , mols = self.compute_pareto_front()
                best_loss = np.argsort(front)
                if verbose:
                    print("Best Candidates:")
                    [print(m,  "{:.2f}".format(P1)+" eV") for m,P1 in zip(self.pareto_mols[best_loss], front[best_loss])]
                f = open(filename+".dat", "w")
                f.write("Best Candidates:\n")
                [f.write(m+" "+"{:.2f}".format(P1)+" eV\n") for m,P1 in zip(self.pareto_mols[best_loss], front[best_loss])]
                f.close()

        except:
            print("Currently only one or two properties are supported.")
            raise ValueError("Wrong shape of values array")


    def plot_pateto_front(self, name):


        try:
            if self.values.shape == (len(self.values),3):
                print("Plot Pareto Front 2d")
                fig,ax1= plt.subplots(figsize=(8,8))

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


                p1  = self.values[:,0]
                p2  = self.values[:,1]
                summe = self.values[:,2]
                xi = np.linspace(min(p1), max(p1), 100)
                yi = np.linspace(min(p2), max(p2), 100)

                triang = tri.Triangulation(p1, p2)
                interpolator = tri.LinearTriInterpolator(triang,summe)
                Xi, Yi = np.meshgrid(xi, yi)
                zi = interpolator(Xi, Yi)

                ax1.contour(xi, yi, zi, levels=18, linewidths=0.5, colors='k')
                sc = ax1.scatter(p1, p2,s =4, c=summe)
                plt.xlabel("$E_{\\rm {at}}/ N^{\\rm tot}$"  + " [eV]", fontsize=21)
                plt.ylabel("$E_{\\rm {gap}}$" + " [eV]", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
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

                plt.plot(self.pareto_front[:,0],self.pareto_front[:,1],'o', color='black')
                plt.plot(self.pareto_front[:,0],self.pareto_front[:,1],'k-',linewidth=2)
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.savefig("{}_pareto.pdf".format(name))
                plt.close()

            elif self.values.shape == (len(self.values),):
                print("Plot Pareto Front 1d")

        except:
            print("Currently only one or two properties are supported.")
            raise ValueError("Wrong shape of values array")

    def plot_chem_space(self, name):
        
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

        print("Compute PCA")
        X_2d, reducer = self.compute_PCA()
        target = self.target
        X_target = rdkit_descriptors.get_all_FP([target], fp_type=self.fp_type)
        X_extra = reducer.transform(X_target)

        print("Plot PCA")
        if self.values.shape == (len(self.values),3):
            sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=200,alpha=0.1,marker="o", c=self.values[:,2],edgecolors='none')
        if self.values.shape == (len(self.values),2):
            ax2.scatter(x=X_extra[:,0], y=X_extra[:,1],s=200,marker="x",edgecolors='none')
            sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=200,alpha=0.5,marker="o", c=self.values[:,1],edgecolors='none')
        if self.values.shape == (len(self.values),):
            sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=200,alpha=0.1,marker="o", c=self.values,edgecolors='none')
        ax2.set_xlabel("PC1", fontsize=21)
        ax2.set_ylabel("PC2", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)

        clb = plt.colorbar(sc)
        clb.set_label('Loss')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("{}_PCA.pdf".format(name))
        plt.close()