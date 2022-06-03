# If we explore diatomic molecule graph, this function will create  
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

class QM9_properties:

    """
    Interface for QM9 property prediction, uses RdKit features
    that can be extracted only from a molecular graph
    model_path : Path to the QM9 machine created with train_qm9.py
    """

    def __init__(self, model_path):
        import joblib
        self.scaler_function = joblib.load(model_path+"scaler.sav")
        self.ml_model = joblib.load(model_path+"ATOMIZATION")
                                                       

    def __call__(self, trajectory_point_in):
        from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit   
        import deepchem as dc
    
        _, _, _, canon_SMILES = chemgraph_to_canonical_rdkit(
            trajectory_point_in.egc.chemgraph)
        print(canon_SMILES)
        X_test = dc.feat.RDKitDescriptors(
            ipc_avg=True).featurize([canon_SMILES])
        prediction = self.scaler_function.inverse_transform(self.ml_model.predict(X_test).reshape(-1,1))
        return prediction[-1]

class multi_obj:

    """
    Combine multiple minimize functions in various different ways.
    Adjust weights for each property necessary because properties live on different orders 
    of magnitude. Clever way might be to normalize each property by the initial
    value at the fist point of the trajectory

    fct_list    : List of minimized functions
    fct_weights : Weights between minimized functions, len(fct_weights) == len(fct_list)
    """

    def __init__(self, fct_list, fct_weights, init_gc, normalize=True):

        self.fct_list   = fct_list
        self.fct_weights = fct_weights
        self.normalize = normalize
        self.init_gc    = init_gc


        """
        trajectory_point_in must be initial cg!!!
        but getting an error with how i implemented it below
        if self.normalize:
            self.fct_initial = []
            for fct in zip(self.fct_list):
                #print(self.fct_list)
                #print(fct)
                print(self.init_gc)
                print(fct[0].__call__(self.init_gc[0]))
                self.fct_initial.append(fct[0].__call__(self.init_gc))
                #print(fct[0].__call__(self.init_gc))
        """

    def __call__(self,trajectory_point_in):
        

        s = 0
        for fct, w in zip(self.fct_list,self.fct_weights): 
            
            """
            uncomment as soon as there is a way to get the initial values
            of each property. 
            Possible evaluate the properties in parallel to make it faster
            """

            #, self.fct_initial):
            #value = (w/abs(fct_init)) * fct.__call__(trajectory_point_in)

            value = w * fct.__call__(trajectory_point_in)
            s+=value

        return s
