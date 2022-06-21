
from bmapqml.utils import * 

class analyze_random_walk:

    """
    Class for analyzing a random walk after the simulation.
    Visualize chemical space and convex hull of the walk.
    """

    def __init__(self, histogram, model=None, verbose=False):
        import numpy as np
        
        """
        histogram : list of all unique encountered points in the random walk.
        minimize_fct : if desired reevaluate all unique molecules using this function
        """

        import pickle
        self.tps     = pickle.load(open(histogram, "rb"))
        self.histogram= np.array(self.convert_to_smiles(pickle.load(open(histogram, "rb"))))
        self.model = None or model
        if verbose:
            print("Number of unique points in the random walk:", len(self.tps))


        """
        self.saved_candidates=pickle.load(open(saved_candidates, "rb"))
        self.saved_candidates_tps, self.saved_candidates_func_val = [],[]
        for mol in self.saved_candidates:
            self.saved_candidates_tps.append(mol.tp)
            self.saved_candidates_func_val.append(mol.func_val)
        del(self.saved_candidates)
        self.saved_candidates = self.convert_to_smiles(self.saved_candidates_tps)
        self.saved_candidates_func_val = np.array(self.saved_candidates_func_val)
        """
        

    
    def convert_to_smiles(self, mols):

        """
        Convert the list of molecules to SMILES strings.
        """

        import rdkit
        from rdkit import Chem

        """
        tp_list: list of molecules as trajectory points
        smiles_mol: list of rdkit molecules
        """

        smiles_mol = []
        for tp in mols:
            
            rdkit_obj = trajectory_point_to_canonical_rdkit(tp)
            smiles_mol.append(rdkit_obj[3])

        return smiles_mol


    def compute_representations(self):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        X = rdkit_descriptors.get_all_FP(self.histogram, fp_type="both")
        return X


    def compute_PCA(self):

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = self.compute_representations()
        reducer = PCA(n_components=2)
        reducer.fit(X)
        X_2d = reducer.transform(X)
        scaler = StandardScaler()
        X_2d = scaler.fit_transform(X_2d)
        return X_2d




    def evaluate_histogram(self, save_histogram=False):
        import numpy as np
        
        """
        Compute values of a function evaluated on all unique smiles in the random walk.
        """

        self.values = self.model.evaluate_trajectory(self.tps)

        if save_histogram:
            np.savez_compressed("QM9_histogram_values", values=self.values)

        return self.values

    def compute_pareto_front(self):
        
        """
        values: array of function values
        Returns values of points in the pareto front.
        pareto_front as well as the indices
        """

            #except:
            #    self.values = np.load("QM9_histogram_values.npz")["values"]
            #   two_dim = self.values[:,:2]

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

        import numpy as np

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
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

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
        import matplotlib.pyplot as plt
        
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
        X_2d = self.compute_PCA()
        print("Plot PCA")
        if self.values.shape == (len(self.values),3):
            sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=400,alpha=0.1,marker="o", c=self.values[:,2],edgecolors='none')
        if self.values.shape == (len(self.values),):
            sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=400,alpha=0.1,marker="o", c=self.values,edgecolors='none')
        ax2.set_xlabel("PC1", fontsize=21)
        ax2.set_ylabel("PC2", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)

        clb = plt.colorbar(sc)
        clb.set_label('Loss')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("{}_PCA.pdf".format(name))
        plt.close()



    def moltosvg(self, mol,molSize=(450,15),kekulize=True):
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem.Draw import rdDepictor

        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:','')


    def interactive_chem_space(self):
        
        """
        Visualize the chemical space in an interactive plot combining the
        pareto front and lewis structures of molecules
        """

        try:

            import numpy as np
            import matplotlib.pyplot as plt
            import mpld3
            from rdkit.Chem.Draw import IPythonConsole
            from rdkit.Chem import Draw
            from rdkit.Chem import AllChem
            from mpld3 import plugins
            mpld3.enable_notebook()
        
        except:
            print("Please install the following packages:")
            print("matplotlib, mpld3, rdkit, seaborn")
            raise ValueError("Missing packages")
        
        svgs = [self.moltosvg(m) for m in self.pareto_mols]
        fig3,ax3= plt.subplots(figsize=(8,8))
        points = ax3.scatter(self.pareto_front[:,0],self.pareto_front[:,1]) 
        tooltip = plugins.PointHTMLTooltip(points, svgs)
        plugins.connect(fig3, tooltip)
