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
        X_test = dc.feat.RDKitDescriptors(
            ipc_avg=True).featurize([canon_SMILES])
        prediction = self.scaler_function.inverse_transform(self.ml_model.predict(X_test).reshape(-1,1))
        return prediction[-1]