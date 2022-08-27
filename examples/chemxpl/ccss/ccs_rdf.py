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

matplotlib.use('Agg')


def mc_run(init_cg, min_func, label, respath):
    """
    Perform as MC simulation
    """

    min_func_name = "chemspacesampler"
    seed = int(str(hash(label))[1:8])
    np.random.seed(1337+seed)
    random.seed(1337+seed)

    possible_elements=["C", "O", "N", "F"]
    forbidden_bonds=None
    ref_beta=4000
    bias_coeff=0.25
    vbeta_bias_coeff=1.e-5
    bound_enforcing_coeff=1.0
    betas=[None, None, ref_beta, ref_beta/2, ref_beta/4, ref_beta/8]

    

    randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [1,30], "final_nhatoms_range": [1, 30],
                                "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                                "forbidden_bonds": forbidden_bonds}
    global_change_params     = {"num_parallel_tempering_tries" : 10, "num_genetic_tries" : 10, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}

    num_MC_steps = 5000


    negcs=len(betas)
    init_egcs =[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

    rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                                bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                                init_egcs=init_egcs, conserve_stochiometry=False, min_function_name=min_func_name,
                                keep_histogram=True, num_saved_candidates=4, vbeta_bias_coeff=vbeta_bias_coeff, keep_full_trajectory=True, restart_file=respath+f"/{label}.pkl")


    for MC_step in range(num_MC_steps):
        rw.global_random_change(**global_change_params)

    traj=rw.ordered_trajectory()
    rw.make_restart(tarball=True)




class RDF_Plotter:

    def __init__(self,d_range,target,molecule,n_shuffle=1,nBits=2048, ncpu=6, verbose=True):

        self.target    = target
        self.molecule = molecule
        self.nBits = nBits
        self.verbose = verbose
        self.n_shuffle = n_shuffle
        self.ncpu  = ncpu

        self.min_func_name, self.fp_type = "chemspacesampler", "MorganFingerprint"
        self.X_init = rdkit_descriptors.get_all_FP([self.target],self.fp_type, nBits=self.nBits) 

        self.d_range = d_range
        
        assert len(self.d_range) == 3
        assert self.d_range[0] < self.d_range[1]
    
        self.min_d, self.max_d, self.d_int = self.d_range[0], self.d_range[1], self.d_range[2]
        self.D_HIGH = np.arange(self.min_d, self.max_d, self.d_int)


    def elbow_curve(self, x, a, b):
        return a * (x/(b+x))


    def fit_convergence(self, N_ALL):

        N_EXT = []
        for n_curr in N_ALL:
            if np.isinf(n_curr).any() == False:
                popt, _ = curve_fit(self.elbow_curve,np.arange(len(n_curr)) ,n_curr, p0=[n_curr[-1], 1])
                print(popt)
                a = popt[0]
                if a < 90000:
                    N_EXT.append(a)
                else:
                    N_EXT.append(np.nan)

            else:
                N_EXT.append(np.nan)

        N_EXT = np.array(N_EXT)

        return N_EXT


    def plot_rdf(self, N_SAMPLES,N_MEAN,N_EXT):

        D_LOW    = np.array([dh-self.d_int for dh in self.D_HIGH])
        D_CENTER = np.array([(dl+dh)/2 for dl, dh in zip(D_LOW,self.D_HIGH )] )

        N_LOWER  = N_SAMPLES[0][:,-1]
    
    
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

        fig, ax1 = plt.subplots(1, 1,figsize=(16,8))
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

        ax1.set_xlim(2.2,self.D_HIGH[-1])
        ax1.plot(D_CENTER, N_LOWER,"-o",color="black", label=self.molecule )
        ax1.plot(D_CENTER, N_EXT,"+",ms=50,c="black" , label=self.molecule )
        ax1.set_yscale("log")
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(f"{self.molecule}.pdf")



    def inner_loop(self, ALL_HISTOGRAMS, i1,i2, ana):
        random.shuffle(ALL_HISTOGRAMS)
        dh = self.D_HIGH[i1]
        dl = dh-self.d_int  
        
        curr_h = pd.concat(ALL_HISTOGRAMS[:i2])
        curr_h = curr_h.drop_duplicates(subset=['SMILES'])
        SMILES = curr_h.values.flatten()
        N, logRDF = ana.count_shell(self.X_init, SMILES, dl, dh, nBits=self.nBits)
        
        return N, logRDF
        
    def process_RDF(self):
        N_ALL, LOG_RDF = [], []
        for i1 in range(len(self.D_HIGH)):
            print(f"Distance {i1}")
            ana   = Analyze(f"/store/jan/trash/mc_sampling/rdf/{self.molecule}_{i1}_*.pkl", full_traj=False, verbose=True)
            ALL_HISTOGRAMS, _,_ = ana.parse_results()

            rdf   = Parallel(n_jobs=self.ncpu, verbose=50)(delayed(self.inner_loop)(ALL_HISTOGRAMS,i1, i2, ana) for i2 in range(1, len(ALL_HISTOGRAMS)+1))
            n_curr, log_rdf_curr = zip(*rdf)
            n_curr, log_rdf_curr = np.array(n_curr), np.array(log_rdf_curr)

            N_ALL.append(n_curr)
            LOG_RDF.append(log_rdf_curr)

        N_ALL, LOG_RDF = np.array(N_ALL), np.array(LOG_RDF)
        return N_ALL, LOG_RDF


    def sample_convergence_RDF(self):
        
        N_SAMPLES, LOG_RDF_SAMPLES = [], []
        for s in range(self.n_shuffle):
            print(f"Sample {s}")
            N_ALL, LOG_RDF = self.process_RDF()
            N_SAMPLES.append(N_ALL)
            LOG_RDF_SAMPLES.append(LOG_RDF)


        N_SAMPLES,LOG_RDF_SAMPLES = np.array(N_SAMPLES), np.array(LOG_RDF_SAMPLES)
        N_MEAN, LOG_RDF_MEAN = np.nanmean(N_SAMPLES, axis=0), np.nanmean(LOG_RDF_SAMPLES, axis=0)


        N_EXT = []
        for i1 in range(len(self.D_HIGH)):
            N_EXT.append(self.fit_convergence([N_MEAN[i1]]))

        N_EXT = np.array(N_EXT).flatten()

        return N_SAMPLES,LOG_RDF_SAMPLES , N_MEAN,N_EXT, LOG_RDF_MEAN