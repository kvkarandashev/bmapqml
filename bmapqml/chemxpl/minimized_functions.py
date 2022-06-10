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


from .utils import chemgraph_to_canonical_rdkit   

def trajectory_point_to_canonical_rdkit(tp_in):
    return chemgraph_to_canonical_rdkit(tp_in.egc.chemgraph)


class QM9_properties:

    """
    Interface for QM9 property prediction, uses RdKit features
    that can be extracted only from a molecular graph
    model_path : Path to the QM9 machine created with train_qm9.py
    """

    def __init__(self, model_path, verbose=False):
        import pickle
        #from bmapqml import examples
        from examples.chemxpl.rdkit_tools import rdkit_descriptors

        self.ml_model = pickle.load(open(model_path+"KRR_1024_atomization", "rb"))
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

