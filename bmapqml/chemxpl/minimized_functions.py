# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
import numpy as np
from numpy.linalg import norm
import pickle
from .utils import (
    trajectory_point_to_canonical_rdkit,
    chemgraph_to_canonical_rdkit,
    coord_info_from_tp,
    MMFFInconsistent,
)
from ..utils import read_xyz_file
from joblib import Parallel, delayed


class Diatomic_barrier:
    def __init__(self, possible_nuclear_charges):
        self.larger_nuclear_charge = max(possible_nuclear_charges)

    def __call__(self, trajectory_point_in):
        cg = trajectory_point_in.egc.chemgraph
        return self.ncharge_pot(cg) + self.bond_pot(cg)

    def ncharge_pot(self, cg):
        if cg.hatoms[0].ncharge == cg.hatoms[1].ncharge:
            if cg.hatoms[0].ncharge == self.larger_nuclear_charge:
                return 1.0
            else:
                return 0.0
        else:
            return 2.0

    def bond_pot(self, cg):
        return float(cg.bond_order(0, 1) - 1)


class OrderSlide:
    def __init__(self, possible_nuclear_charges_input):
        possible_nuclear_charges = sorted(possible_nuclear_charges_input)
        self.order_dict = {}
        for i, ncharge in enumerate(possible_nuclear_charges):
            self.order_dict[ncharge] = i

    def __call__(self, trajectory_point_in):
        return sum(
            self.order_dict[ha.ncharge]
            for ha in trajectory_point_in.egc.chemgraph.hatoms
        )


from .rdkit_descriptors import extended_get_single_FP


class RepGenFuncProblem(Exception):
    pass


class InterfacedModel:
    def __init__(self, model):
        """
        Returns prediction of the KRR model for a TrajectoryPoint object.
        """
        self.model = model

    def __call__(self, trajectory_point_in):
        try:
            representation = self.representation_func(trajectory_point_in)
            return self.model(representation)
        except RepGenFuncProblem:
            return None


class MMFF_based_model(InterfacedModel):
    def __init__(self, *args, **kwargs):
        """
        Prediction of a model for TrajectoryPoint object that uses MMFF coordinates as input.
        """
        super().__init__(*args, **kwargs)

        self.add_info_dict = {"coord_info": coord_info_from_tp}

    def representation_func(self, trajectory_point_in):
        coordinates = trajectory_point_in.calc_or_lookup(self.add_info_dict)[
            "coord_info"
        ]["coordinates"]
        if coordinates is None:
            raise RepGenFuncProblem
        nuclear_charges = trajectory_point_in.egc.true_ncharges()
        return self.coord_representation_func(coordinates, nuclear_charges)


class SLATM_MMFF_based_model(MMFF_based_model):
    """
    An inheritor to MMFF_based_model that uses SLATM representation.
    """

    def __init__(self, *args, mbtypes=None, **other_kwargs):
        from qml.representations import generate_slatm

        super().__init__(*args, **other_kwargs)
        self.mbtypes = mbtypes
        self.rep_func = generate_slatm

    def verify_mbtypes(self, xyz_files):
        from qml.representations import get_slatm_mbtypes

        all_nuclear_charges = []
        max_natoms = 0
        for f in xyz_files:
            (
                nuclear_charges,
                _,
                _,
                _,
            ) = read_xyz_file(f)
            max_natoms = max(len(nuclear_charges), max_natoms)
            all_nuclear_charges.append(nuclear_charges)
        np_all_nuclear_charges = np.zeros(
            (len(all_nuclear_charges), max_natoms), dtype=int
        )
        for mol_id, nuclear_charges in enumerate(all_nuclear_charges):
            cur_natoms = len(nuclear_charges)
            np_all_nuclear_charges[mol_id, :cur_natoms] = nuclear_charges[:]
        self.mbtypes = get_slatm_mbtypes(np_all_nuclear_charges)

    def coord_representation_func(self, coordinates, nuclear_charges):
        return self.rep_func(coordinates, nuclear_charges, self.mbtypes)


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

    def __call__(self, trajectory_point_in):
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

        X_test = extended_get_single_FP(canon_SMILES, "both")
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

        # Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=4)(
            delayed(self.__call__)(tp_in) for tp_in in trajectory_points
        )
        return np.array(values)


class find_match:

    """
    Guide Random walk towards the target the representation corresponding to the target
    molecule.
    """

    def __init__(self, X_target, verbose=False):

        self.X_target = X_target
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

    def __call__(self, trajectory_point_in):

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, "both")
        d = norm(X_test - self.X_target)

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ", d)

            return np.exp(d)


class sample_local_space:
    """
    Sample local chemical space of the inital compound
    with input representation X_init. Radial symmetric
    LJ12-6 potential is used. Epsilon and sigma parameters
    can be changed to adjust the potential and how different
    the target molecule can be from the initial one.
    """

    def __init__(
        self,
        X_init,
        verbose=False,
        pot_type="harmonic",
        epsilon=1.0,
        sigma=1.0,
        gamma=1,
    ):

        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.X_init = X_init
        self.pot_type = pot_type
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

    def lennard_jones_potential(self, d):
        """
        Lennard-Jones potential, d is the distance between the two points
        in chemical space. The potential is given by:
        """
        return 4 * self.epsilon * ((self.sigma / d) ** 12 - (self.sigma / d) ** 6)

    def harmonic_potential(self, d):
        """
        Flat-bottomed harmonic potential. within the range of 0 to sigma
        the potential is flat and has value epsilon. Outsite it is quadratic.
        This avoids the problem of the potential being infinite at zero such that
        also molecule close to or equal to d = 0 can be sampled
        """

        if d <= self.sigma:
            return -self.epsilon  # *100
        else:
            return (d - self.sigma) ** 2 - self.epsilon

    def buckingham_potential(self, d):
        """
        Returns the buckingham potential for a given distance.
        It is a combination of an exponential and a one over 6 power
        using the parameters epsilon and sigma and gamma.
        """

        return self.epsilon * np.exp(-self.sigma * d) - self.gamma * (1 / d**6)

    def exponential_potential(self, d):
        """
        Returns the exponential potential for a given distance.
        """

        return self.epsilon * np.exp(-self.sigma * d)

    def __call__(self, trajectory_point_in):

        # KK: This demonstrates how expensive intermediate data can be saved too.
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, "both")

        d = norm(X_test - self.X_init)
        if self.pot_type == "lj":
            V = self.lennard_jones_potential(d)
        elif self.pot_type == "harmonic":
            V = self.harmonic_potential(d)
        elif self.pot_type == "buckingham":
            V = self.buckingham_potential(d)
        elif self.pot_type == "exponential":
            V = self.exponential_potential(d)

        if self.verbose:
            print("SMILE:", canon_SMILES, d, V)

        return V

    def evaluate_point(self, trajectory_point_in):
        """
        Evaluate the function on a list of trajectory points
        """
        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, "both")
        d = norm(X_test - self.X_init)

        if self.pot_type == "lj":
            V = self.lennard_jones_potential(d)
        elif self.pot_type == "harmonic":
            V = self.harmonic_potential(d)
        elif self.pot_type == "buckingham":
            V = self.buckingham_potential(d)
        elif self.pot_type == "exponential":
            V = self.exponential_potential(d)

        return V, d

    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points
        """

        # Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=24)(
            delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points
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

        # from joblib import Parallel, delayed
        # values = Parallel(n_jobs=2)(delayed(fct.__call__)(trajectory_point_in) for fct in self.fct_list)

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

        # Aparently this is the fastest way to do this:
        values = Parallel(n_jobs=4)(
            delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points
        )
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

    def __init__(self, model_path, rdkit_property, max=True, verbose=True):

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
