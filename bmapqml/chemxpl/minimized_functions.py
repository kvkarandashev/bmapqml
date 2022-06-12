# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
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
    """

    def __init__(self, model_path, verbose=False):
        import pickle
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        from examples.chemxpl.rdkit_tools import rdkit_descriptors

        self.ml_model = pickle.load(open(model_path, "rb"))
        self.verbose  = verbose
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}

    def __call__(self, trajectory_point_in):
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
        from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit   

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        X_test = rdkit_descriptors.get_all_FP([canon_SMILES], fp_type="both")
        prediction = self.ml_model.predict(X_test.reshape(1, -1))

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ", prediction[0])
        return prediction[-1]

class multi_obj:

    """
    Combine multiple minimize functions in various different ways.
    Adjust weights for each property necessary because properties live on different orders 
    of magnitude. Clever way might be to use approximate average values of these properties in the 
    chemical space of interest

    Average values in QM9
     6.8  eV for band gap
    -1.9  eV for atomization energy

    fct_list    : List of minimized functions
    fct_weights : Weights between minimized functions, len(fct_weights) == len(fct_list)
    """


    def __init__(self, fct_list, fct_weights, verbose=False):
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        self.fct_list   = fct_list
        self.fct_weights = fct_weights
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}
        self.verbose = verbose

    def __call__(self,trajectory_point_in):
        import numpy as np
        #from joblib import Parallel, delayed

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(self.canonical_rdkit_output)["canonical_rdkit"]

        s = 0

        #values = Parallel(n_jobs=2)(delayed(fct.__call__)(trajectory_point_in) for fct in self.fct_list)
        values = []

        for fct in self.fct_list: 

            values.append(fct.__call__(trajectory_point_in))

        values = np.array(values)

        s = np.dot(self.fct_weights, values)


        if self.verbose:
            print("SMILE:", canon_SMILES, "v1", values[0],"v2", values[1])

        return s



class Rdkit_properties:
    def __init__(self,model_path, rdkit_property, max=True, verbose=True):
        from bmapqml.utils import trajectory_point_to_canonical_rdkit
        self.rdkit_property=rdkit_property
        self.canonical_rdkit_output={"canonical_rdkit" : trajectory_point_to_canonical_rdkit}
        self.max = max
        self.verbose = verbose

    def __call__(self, trajectory_point_in):
        import numpy as np
        from examples.chemxpl.rdkit_tools import rdkit_descriptors
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



