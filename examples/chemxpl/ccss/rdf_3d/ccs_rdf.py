from bmapqml.chemxpl.plotting import Analyze
import matplotlib
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt
from bmapqml.chemxpl import rdkit_descriptors
import pandas as pd
from joblib import Parallel, delayed
import random 
import numpy as np
from scipy.optimize import curve_fit
from bmapqml.chemxpl.random_walk import RandomWalk
from bmapqml.chemxpl import ExtGraphCompound
from copy import deepcopy
import pdb
matplotlib.use('Agg')


def mc_run(init_cg, min_func, label, respath,betas=[None, None, 8000, 8000/2, 8000/4, 8000/8],num_MC_steps = 10000,bias_coeff=0.25,vbeta_bias_coeff=1.e-5,bound_enforcing_coeff=1.0):
    
    """
    Perform as MC simulation
    """

    min_func_name = "chemspacesampler"
    seed = int(str(hash(label))[1:8])
    np.random.seed(1337+seed)
    random.seed(1337+seed)

    possible_elements=["C", "O", "N", "F"]
    forbidden_bonds=None
    
    bias_coeff=0.25
    vbeta_bias_coeff=1.e-5
    bound_enforcing_coeff=1.0

    randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [1,30], "final_nhatoms_range": [1, 30],
                                "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                                "forbidden_bonds": forbidden_bonds}
    global_change_params     = {"num_parallel_tempering_tries" : 10, "num_genetic_tries" : 10, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}

    


    negcs=len(betas)
    init_egcs =[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

    rw=RandomWalk(bias_coeff=bias_coeff,randomized_change_params=randomized_change_params,
                                bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                                init_egcs=init_egcs, conserve_stochiometry=False, min_function_name=min_func_name,
                                keep_histogram=True,linear_storage=True, num_saved_candidates=4, vbeta_bias_coeff=vbeta_bias_coeff, keep_full_trajectory=True, restart_file=respath+f"/{label}.pkl")


    for MC_step in range(num_MC_steps):
        #print(f"MC step {MC_step}")
        rw.global_random_change(**global_change_params)

    rw.ordered_trajectory()
    rw.make_restart(tarball=True)


class RDF_Plotter:

    def __init__(self,respath,d_range,target,molecule,n_shuffle=1,nBits=2048, ncpu=6, verbose=True, epsilon=5.0):
        self.respath = respath
        self.target    = target
        self.molecule = molecule
        self.nBits = nBits or 2048
        self.verbose = verbose
        self.n_shuffle = n_shuffle
        self.ncpu  = ncpu

        self.min_func_name, self.fp_type = "chemspacesampler", "MorganFingerprint"
        self.X_init = rdkit_descriptors.get_all_FP([self.target],self.fp_type, nBits=self.nBits) 

        self.d_range = d_range
        self.epsilon = epsilon
        assert len(self.d_range) == 3
        assert self.d_range[0] < self.d_range[1]
    
        self.min_d, self.max_d, self.d_int = self.d_range[0], self.d_range[1], self.d_range[2]
        self.D_HIGH = np.arange(self.min_d, self.max_d, self.d_int)


    def elbow_curve(self, x, a, b): return a * (x/(b+x))


    def fit_convergence(self, N_ALL, sigma=None):

        N_EXT = []
        pdb.set_trace()
        for n_curr in N_ALL:
            if np.isinf(n_curr).any() == False:
                if np.any(sigma==None):
                    popt, pcov = curve_fit(self.elbow_curve,np.arange(len(n_curr)) ,n_curr, p0=[n_curr[-1], (1+n_curr[-1]/2)],bounds=((0,0), (np.inf,np.inf)))
                else:
                    popt, pcov = curve_fit(self.elbow_curve,np.arange(len(n_curr)) ,n_curr, p0=[n_curr[-1], (1+n_curr[-1]/2)],bounds=((0,0), (np.inf,np.inf)),sigma=sigma)
                perr = np.sqrt(np.diag(pcov))[0]
                a = popt[0]
                print(a, perr)
                if a < 1e15:
                    N_EXT.append([a, perr])
                else:
                    N_EXT.append(np.nan)

            else:
                N_EXT.append(np.nan)

        return N_EXT


    def plot_rdf(self, N_SAMPLES,N_MEAN,N_EXT):

        D_LOW    = np.array([dh-self.d_int for dh in self.D_HIGH])
        D_CENTER = np.array([(dl+dh)/2 for dl, dh in zip(D_LOW,self.D_HIGH )] )
        N_EXT    = N_EXT.reshape(len(D_CENTER),-1)
        #just take the last sample of first run because the final number 
        #of samples is the same for all runs

        N_LOWER = []
        for i in range(len(N_SAMPLES[0])):
            N_LOWER.append(N_SAMPLES[0][i][-1])
    
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("colorblind")


        fs = 26

        plt.rc('font', size=fs)
        plt.rc('axes', titlesize=fs)
        plt.rc('axes', labelsize=fs)           
        plt.rc('xtick', labelsize=fs)          
        plt.rc('ytick', labelsize=fs)          
        plt.rc('legend', fontsize=fs)   
        plt.rc('figure', titlesize=fs) 

        fig, ax1 = plt.subplots(1, 1,figsize=(8,8))
        for a in [ax1]:
            a.spines['right'].set_color('none')
            a.spines['top'].set_color('none')
            a.spines['bottom'].set_position(('axes', -0.05))
            a.spines['bottom'].set_color('black')
            a.spines['left'].set_color('black')
            a.yaxis.set_ticks_position('left')
            a.xaxis.set_ticks_position('bottom')
            a.spines['left'].set_position(('axes', -0.05))  
            a.set_xlabel("$d$")

        
        ax1.set_ylabel("#",rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)

        cmap   = cm.coolwarm(np.linspace(0, 1, N_MEAN.shape[1]))
        for i in range(N_MEAN.shape[1]):
            ax1.plot(D_CENTER, N_MEAN[:,i], color=cmap[i], linewidth=2)

        ax1.plot(D_CENTER, N_LOWER,"-o",color="black", label=self.molecule )

        #[(0,0,0), (5,5,5)]
        ax1.errorbar(D_CENTER,  N_EXT[:,0], yerr= N_EXT[:,1],fmt=',', color="black", label=self.molecule, capsize=5, capthick=2, elinewidth=2, ms=10)
        #matplotlib set lower limit of x axis to 2.1 up to largest point in data
        ax1.set_xlim([2.1, np.max(D_CENTER)+0.1])
        

        
        ax1.set_yscale("log" )
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(f"{self.molecule}.pdf")



    def inner_loop(self, ALL_HISTOGRAMS, i1,i2, ana, return_mols=False):
        random.shuffle(ALL_HISTOGRAMS)
        dh = self.D_HIGH[i1]
        dl = dh-self.d_int  
        
        curr_h = pd.concat(ALL_HISTOGRAMS[:i2])
        curr_h = curr_h.drop_duplicates(subset=['SMILES'])
        #SMILES = curr_h["SMILES"].values.flatten()

        if return_mols==False:
            #N, _ = ana.count_shell(self.X_init, SMILES, dl, dh, nBits=self.nBits, return_mols=False)
            N         = ana.count_shell_value(curr_h,self.epsilon, return_mols=False )
            return N

        else:
            #N, _, SMILES = ana.count_shell(self.X_init, SMILES, dl, dh, nBits=self.nBits, return_mols=True)
            N,MATCHED_SMILES         = ana.count_shell_value(curr_h,self.epsilon, return_mols=True )
            return N, MATCHED_SMILES

        
    def process_RDF(self):
        N_ALL = []
        for i1 in range(len(self.D_HIGH)):
            print(f"Distance {i1}")
            ana   = Analyze(f"{self.respath}"+f"/{self.molecule}_{i1}_*.pkl", full_traj=False, verbose=True, mode="sampling")
            ALL_HISTOGRAMS, _,_ = ana.parse_results()
            
            rdf   = Parallel(n_jobs=self.ncpu, verbose=50)(delayed(self.inner_loop)(ALL_HISTOGRAMS,i1, i2, ana, False) for i2 in range(1, len(ALL_HISTOGRAMS)+1))
            rdf = np.array(rdf)
            N_ALL.append(rdf)

        N_ALL = np.array(N_ALL)
        return N_ALL


    def get_samples(self):
        D_ALL = []
        D_MOL_SAMPLES = []
        for i1 in range(len(self.D_HIGH)):
            print(f"Distance {i1}")
            ana   = Analyze(f"{self.respath}"+f"/{self.molecule}_{i1}_*.pkl", full_traj=False, verbose=True,  mode="sampling")
            ALL_HISTOGRAMS, _,_ = ana.parse_results()
            N,MATCHED_SMILES = self.inner_loop(ALL_HISTOGRAMS,i1, len(ALL_HISTOGRAMS), ana, True)
            
            if len(MATCHED_SMILES)>0:
                D_ALL.append([self.D_HIGH[i1]]*len(MATCHED_SMILES))
                D_MOL_SAMPLES.append(MATCHED_SMILES)
            else:
                D_ALL.append([self.D_HIGH[i1]])
                D_MOL_SAMPLES.append([np.nan])

        return D_ALL, D_MOL_SAMPLES


    def sample_convergence_RDF(self):
        
        N_SAMPLES = []
        for s in range(self.n_shuffle):
            print(f"Sample {s}")
            N_ALL = self.process_RDF()
            N_SAMPLES.append(N_ALL)


        N_EXT, N_MEAN,N_STD = [], [], []
        N_SAMPLES = np.array(N_SAMPLES)
        N_STD = np.zeros_like(N_SAMPLES)
        try:
            N_MEAN = np.nanmean(N_SAMPLES, axis=0)
            N_STD  = np.nanstd(N_SAMPLES, axis=0)
            
            N_STD[:,-1] = N_STD[:,-2]
            #pdb.set_trace()
            for i1 in range(len(self.D_HIGH)):
                #to avoid division by zero
                N_STD[i1][N_STD[i1] == 0] = 1
                #pdb.set_trace()
                N_EXT.append(self.fit_convergence([N_MEAN[i1]],sigma=N_STD[i1] ))

            N_EXT = np.array(N_EXT) #.flatten()
            print(N_SAMPLES.shape)
            print("N_EXT", N_EXT.shape)
            
            return N_SAMPLES , N_MEAN,N_STD, N_EXT

        except:
            print(N_SAMPLES.shape)
            return N_SAMPLES , N_MEAN,N_STD,N_EXT
        

def loadXYZ(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        numAtoms = int(lines[0])
        positions = np.zeros((numAtoms, 3), dtype=np.double)
        elems = [None] * numAtoms
        comment = lines[1]
        for x in range (2, 2 + numAtoms):
            line_split = lines[x].rsplit()
            elems[x - 2] = line_split[0]
            
            line_split[1] = line_split[1].replace('*^', 'E')
            line_split[2] = line_split[2].replace('*^', 'E')
            line_split[3] = line_split[3].replace('*^', 'E')
            
            positions[x - 2][0] = np.double(line_split[1]) 
            positions[x - 2][1] = np.double(line_split[2]) 
            positions[x - 2][2] = np.double(line_split[3])
                
    return np.asarray(elems), np.asarray(positions), comment