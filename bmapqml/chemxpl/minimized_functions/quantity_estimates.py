from .. import rdkit_descriptors
from rdkit import Chem
import pickle
from ..utils import trajectory_point_to_canonical_rdkit
from joblib import Parallel, delayed


# TODO Perhaps multi_obj should be combined with this.
class LinearCombination:
    def __init__(self, functions, coefficients, function_names):
        """
        Returns a linear combination of several functions acting on a TrajectoryPoint object.
        functions : functions to be combined
        coefficients : coefficients with which the functions are combined
        function_names : names of the functions, determines the label their values are stored at for TrajectoryPoint object.
        """
        self.coefficients = coefficients
        self.function_names = function_names
        self.function_dict = {}
        for func, func_name in zip(functions, function_names):
            self.function_dict[func_name] = func
        self.call_counter = 0

    def __call__(self, trajectory_point_in):
        self.call_counter += 1
        func_val_dict = trajectory_point_in.calc_or_lookup(self.function_dict)
        output = 0.0
        for coeff, func_name in zip(self.coefficients, self.function_names):
            if func_val_dict[func_name] is None:
                return None
            output += coeff * func_val_dict[func_name]
        return output


class QM9_properties:

    """
    Interface for QM9 property prediction, uses RdKit features
    that can be extracted only from a molecular graph

    model_path : Path to the QM9 machine created with train_qm9.py
    verbose    : If True, prints the SMILE and prediction
    """

    def __init__(self, model_path, max=False, verbose=False):

        self.ml_model = pickle.load(open(model_path, "rb"))
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }
        self.max = max

    def __call__(self, trajectory_point_in):

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, "both")
        prediction = self.ml_model.predict(X_test.reshape(1, -1))

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ", prediction[0])

        if self.max:
            return -prediction[-1]
        else:
            return prediction[-1]

    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points
        """

        values = Parallel(n_jobs=4)(
            delayed(self.__call__)(tp_in) for tp_in in trajectory_points
        )
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

    def __init__(self, fct_list, fct_weights, max=False, verbose=False):

        self.fct_list = fct_list
        self.fct_weights = fct_weights
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }
        self.verbose = verbose
        self.max = max

    def __call__(self, trajectory_point_in):

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        sum = 0
        values = []

        for fct in self.fct_list:

            values.append(fct.__call__(trajectory_point_in))

        values = np.array(values)
        sum = np.dot(self.fct_weights, values)

        if self.verbose:
            if self.max:
                print("SMILE:", canon_SMILES, "v1", -values[0], "v2", -values[1])
            else:
                print("SMILE:", canon_SMILES, "v1", values[0], "v2", values[1])

        return sum

    def evaluate_point(self, trajectory_point_in):

        """
        Evaluate the function on a single trajectory point
        """

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

        values = Parallel(n_jobs=4)(
            delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points
        )
        return np.array(values)


class Rdkit_properties:

    """
    Test function to minimize/maximize a property of a molecule given by a SMILES string which can be
    evaluated directly using rdkit. This does not use any machine learning models and is only for testing
    E.g. if max=True and rdkit_property=rdkit.descriptors.NumRotatableBonds the algorithm should maximize the
    number of rotatible bonds of the molecule. This can be checked by running the simulation.
    rdkit_property : rdkit property to minimize/maximize
    verbose : print out information about the molecule
    """

    def __init__(self, rdkit_property, max=True, verbose=True):

        self.rdkit_property = rdkit_property
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }
        self.max = max
        self.verbose = verbose

    def __call__(self, trajectory_point_in):

        fct = self.rdkit_property
        rdkit_mol, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]
        rdkit_mol = Chem.AddHs(rdkit_mol)
        value = fct(rdkit_mol)

        if self.verbose:
            print(canon_SMILES, value)

        if self.max:
            return np.exp(-value)
        else:
            return value
