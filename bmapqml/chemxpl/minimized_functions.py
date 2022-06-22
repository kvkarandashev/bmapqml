# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
from tabnanny import verbose


class Diatomic_barrier:
    def __init__(self, possible_nuclear_charges):
        self.larger_nuclear_charge=max(possible_nuclear_charges)
    def __call__(self, trajectory_point_in):
        cg=trajectory_point_in.egc.chemgraph
        return self.ncharge_pot(cg)+self.bond_pot(cg)
    def ncharge_pot(self, cg):
        if cg.hatoms[0].ncharge==cg.hatoms[1].ncharge:
            if cg.hatoms[0].ncharge==self.larger_nuclear_charge:
                return 1.
            else:
                return .0
        else:
            return 2.
    def bond_pot(self, cg):
        return float(cg.bond_order(0, 1)-1)

class OrderSlide:
    def __init__(self, possible_nuclear_charges_input):
        possible_nuclear_charges=sorted(possible_nuclear_charges_input)
        self.order_dict={}
        for i, ncharge in enumerate(possible_nuclear_charges):
            self.order_dict[ncharge]=i
    def __call__(self, trajectory_point_in):
        return sum(self.order_dict[ha.ncharge] for ha in trajectory_point_in.egc.chemgraph.hatoms)



class QM9_properties:

    """
    Interface for QM9 property prediction, uses RdKit features
    that can be extracted only from a molecular graph
    
    model_path : Path to the QM9 machine created with train_qm9.py
    verbose    : If True, prints the SMILE and prediction
    """

    def __init__(self, model_path,max=False, verbose=False):
        import pickle
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        from examples.chemxpl.rdkit_tools import rdkit_descriptors

        self.ml_model = pickle.load(open(model_path, "rb"))
        self.verbose  = verbose
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}
        self.max=max

    def __call__(self, trajectory_point_in):
        import numpy as np
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit   

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        X_test = rdkit_descriptors.get_all_FP([canon_SMILES], fp_type="both")
        prediction = self.ml_model.predict(X_test.reshape(1, -1))

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ", prediction[0])

        if self.max:
            return np.exp(-prediction)[-1]
        else:
            return prediction[-1]


    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points 
        """
        
        import numpy as np

        from joblib import Parallel, delayed
        #Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=4)(delayed(self.__call__)(tp_in) for tp_in in trajectory_points)
        return np.array(values)


class find_match():

    """
    Guide Random walk towards the target the representation corresponding to the target
    molecule.
    """
    def __init__(self, X_target,verbose=False):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit   
        self.X_target=X_target
        self.verbose = verbose
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}

    def __call__(self, trajectory_point_in):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        import numpy as np
        from numpy.linalg import norm

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        X_test = rdkit_descriptors.get_all_FP([canon_SMILES], fp_type="both")
        d = norm(X_test-self.X_target)

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ",d)

            return np.exp(d)

class sample_local_space():

    """
    Sample local chemical space of the inital compound
    with input representation X_init. Radial symmetric
    LJ12-6 potential is used. Epsilon and sigma parameters
    can be changed to adjust the potential and how different 
    the target molecule can be from the initial one.
    """

    def __init__(self, X_init,verbose=False, epsilon=1., sigma=1.):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit  
        self.epsilon=epsilon
        self.sigma=sigma
        self.X_init=X_init 
        self.verbose = verbose
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}

    def lennard_jones_potential(self,d, epsilon, sigma):
        return 4*epsilon*((sigma/d)**12 - (sigma/d)**6)

    def __call__(self, trajectory_point_in):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        import numpy as np
        from numpy.linalg import norm

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        X_test = rdkit_descriptors.get_all_FP([canon_SMILES], fp_type="both")
        #pdb.set_trace()
        d = norm(X_test-self.X_init)

        V = self.lennard_jones_potential(d, self.epsilon, self.sigma)

        if self.verbose:
            print("SMILE:", canon_SMILES,d, V)

        return V


    def evaluate_point(self, trajectory_point_in):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        import numpy as np
        from numpy.linalg import norm

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        X_test = rdkit_descriptors.get_all_FP([canon_SMILES], fp_type="both")
        d = norm(X_test-self.X_init)

        V = self.lennard_jones_potential(d, self.epsilon, self.sigma)

        return V, d


    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points 
        """
        
        import numpy as np

        from joblib import Parallel, delayed
        #Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=4)(delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points)
        return np.array(values)

    

class multi_obj:

    """
    Combine multiple minimize functions in various different ways.
    Adjust weights for each property necessary because properties live on different orders 
    of magnitude. Clever way might be to use approximate average values of these properties in the 
    chemical space of interest, e.g. :

    Average values in QM9
     6.8  eV for band gap
    -1.9  eV for atomization energy

    fct_list    : List of minimized functions
    fct_weights : Weights between minimized functions, len(fct_weights) == len(fct_list)
    verbose     : Print information about the function and molecule
    """


    def __init__(self, fct_list, fct_weights, verbose=False):
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        self.fct_list   = fct_list
        self.fct_weights = fct_weights
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}
        self.verbose = verbose

    def __call__(self,trajectory_point_in):
        import numpy as np
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        sum = 0
        values = []

        for fct in self.fct_list: 

            values.append(fct.__call__(trajectory_point_in))

        values = np.array(values)

        sum = np.dot(self.fct_weights, values)


        if self.verbose:
            print("SMILE:", canon_SMILES, "v1", values[0],"v2", values[1])

        return sum



    def evaluate_point(self, trajectory_point_in):
        """
        Evaluate the function on a single trajectory point 
        """

        import numpy as np
        #from joblib import Parallel, delayed
        #values = Parallel(n_jobs=2)(delayed(fct.__call__)(trajectory_point_in) for fct in self.fct_list)

        values = []
        for fct in self.fct_list: 

            values.append(fct.__call__(trajectory_point_in))

        sum = np.dot(self.fct_weights, values)
        values.append(sum)
        values = np.array(values)
 
        return values



    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points 
        """
        
        import numpy as np

        from joblib import Parallel, delayed
        #Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=4)(delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points)
        return np.array(values)

        
        """
        #from tqdm import tqdm
        values = []
        for tp_in in tqdm(trajectory_points):
            values.append(self.evaluate_point(tp_in))

        values = np.array(values)

        return values                
    
        """

    



class Rdkit_properties:

    """
    Test function to minimize/maximize a property of a molecule given by a SMILES string which can be
    evaluated directly using rdkit. This does not use any machine learning models and is only for testing
    E.g. if max=True and rdkit_property=rdkit.descriptors.NumRotatableBonds the algorithm should maximize the
    number of rotatible bonds of the molecule. This can be checked by running the simulation.

    model_path : dummy path (not needed)
    rdkit_property : rdkit property to minimize/maximize
    verbose : print out information about the molecule
    """

    def __init__(self,model_path, rdkit_property, max=True, verbose=True):
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        self.rdkit_property=rdkit_property
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}
        self.max = max
        self.verbose = verbose

    def __call__(self, trajectory_point_in):
        import numpy as np
        from rdkit import Chem
    
        fct = self.rdkit_property
        rdkit_mol, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]
        rdkit_mol = Chem.AddHs(rdkit_mol)
        value = fct(rdkit_mol)


        if self.verbose:
            print(canon_SMILES, value)

        if self.max:
            return np.exp(-value)
        else:
            return value





