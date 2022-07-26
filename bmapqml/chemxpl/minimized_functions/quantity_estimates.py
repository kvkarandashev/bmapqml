# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
import numpy as np
import pickle
from ..utils import (
    trajectory_point_to_canonical_rdkit,
    canonical_SMILES_from_tp,
    coord_info_from_tp,
    chemgraph_from_ncharges_coords,
    InvalidAdjMat,
)
from ...utils import read_xyz_file, read_xyz_lines
from joblib import Parallel, delayed
import os, sys
from .. import rdkit_descriptors
from rdkit import Chem


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


class FF_based_model(InterfacedModel):
    def __init__(self, *args, num_ff_attempts=1, ff_type="MMFF", **kwargs):
        """
        Prediction of a model for TrajectoryPoint object that uses FF coordinates as input.
        num_ff_attempts : since FF might not converge the first time, specify how many attempts should be made to see whether it does converge.
        """
        super().__init__(*args, **kwargs)

        self.num_ff_attempts = num_ff_attempts
        self.ff_type = ff_type

        self.add_info_dict = {"coord_info": coord_info_from_tp}
        self.kwargs_dict = {
            "coord_info": {
                "num_attempts": self.num_ff_attempts,
                "ff_type": self.ff_type,
            }
        }

    def representation_func(self, trajectory_point_in):
        coordinates = trajectory_point_in.calc_or_lookup(
            self.add_info_dict, kwargs_dict=self.kwargs_dict
        )["coord_info"]["coordinates"]
        if coordinates is None:
            raise RepGenFuncProblem
        nuclear_charges = trajectory_point_in.egc.true_ncharges()
        return self.coord_representation_func(coordinates, nuclear_charges)


class SLATM_FF_based_model(FF_based_model):
    """
    An inheritor to FF_based_model that uses SLATM representation.
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


# For quantity estimates with xTB and MMFF coordinates.
class LeruliCoordCalc:
    def __init__(self, time_check=1.0e-2):
        from leruli import graph_to_geometry
        from leruli.internal import LeruliInternalError

        self.canon_SMILES_dict = {"canon_SMILES": canonical_SMILES_from_tp}
        self.coord_func = graph_to_geometry
        self.time_check = time_check
        self.time_check_exception = LeruliInternalError

    def __call__(self, tp, **other_kwargs):
        output = {"coordinates": None, "nuclear_charges": None}
        # Create the canonical SMILES string
        canon_SMILES = tp.calc_or_lookup(self.canon_SMILES_dict)["canon_SMILES"]
        coord_info_dict = None
        #        while coord_info_dict is None:
        try:
            coord_info_dict = self.coord_func(canon_SMILES, "XYZ")
            if coord_info_dict is None:
                print("#PROBLEMATIC_SMILES (None returned):", canon_SMILES)
        #                    break
        except self.time_check_exception:
            print("#PROBLEMATIC_SMILES (internal error):", canon_SMILES)
        #                time.sleep(self.time_check)
        if coord_info_dict is None:
            return output
        coord_info_str = coord_info_dict["geometry"]
        nuclear_charges, _, coordinates, _ = read_xyz_lines(coord_info_str.split("\n"))
        # Additionally check that the resulting coordinates actually correspond to the initial chemical graph.
        try:
            cg_from_coords = chemgraph_from_ncharges_coords(
                nuclear_charges, coordinates
            )
            if cg_from_coords == tp.egc.chemgraph:
                output["coordinates"] = coordinates
                output["nuclear_charges"] = nuclear_charges
        except InvalidAdjMat:
            pass
        return output


available_FF_xTB_coord_calculation_types = ["RDKit", "Leruli", "RDKit_wLeruli"]


class FF_xTB_res_dict:
    def __init__(
        self,
        calc_type="GFN2-xTB",
        num_ff_attempts=1,
        ff_type="MMFF",
        coord_calculation_type="RDKit",
        display_problematic_coord_tps=False,
        display_problematic_xTB_tps=False,
    ):
        """
        Calculating xTB result dictionnary produced by tblite library from coordinates obtained by RDKit.
        """
        from tblite.interface import Calculator
        from ..data import conversion_coefficient

        self.calc_type = calc_type
        self.calculator_func = Calculator
        self.call_counter = 0

        self.coord_conversion_coefficient = conversion_coefficient["Angstrom_Bohr"]

        self.num_ff_attempts = num_ff_attempts
        self.ff_type = ff_type
        self.display_problematic_coord_tps = display_problematic_coord_tps
        self.display_problematic_xTB_tps = display_problematic_xTB_tps
        self.coord_calculation_type = coord_calculation_type

        assert self.coord_calculation_type in available_FF_xTB_coord_calculation_types
        if self.coord_calculation_type != "Leruli":
            self.coord_info_from_tp_func = coord_info_from_tp
            if self.coord_calculation_type == "RDKit_wLeruli":
                self.coord_info_from_tp_func_check = LeruliCoordCalc()
        else:
            self.coord_info_from_tp_func = LeruliCoordCalc()

        self.ff_coord_info_dict = {"coord_info": self.coord_info_from_tp_func}
        self.ff_coord_kwargs_dict = {
            "coord_info": {
                "num_attempts": self.num_ff_attempts,
                "ff_type": self.ff_type,
            }
        }

    def __call__(self, tp):
        self.call_counter += 1

        coordinates = tp.calc_or_lookup(
            self.ff_coord_info_dict, kwargs_dict=self.ff_coord_kwargs_dict
        )["coord_info"]["coordinates"]
        nuclear_charges = tp.calc_or_lookup(
            self.ff_coord_info_dict, kwargs_dict=self.ff_coord_kwargs_dict
        )["coord_info"]["nuclear_charges"]

        if self.coord_calculation_type == "RDKit_wLeruli":
            if coordinates is None:
                coord_info = self.coord_info_from_tp_func_check(tp)
                coordinates = coord_info["coordinates"]
                nuclear_charges = coord_info["nuclear_charges"]
                tp.calculated_data["coord_info"] = coord_info
        if coordinates is None:
            if self.display_problematic_coord_tps:
                print("#PROBLEMATIC_COORD_TP:", tp)
            return None
        calc = self.calculator_func(
            self.calc_type,
            nuclear_charges,
            coordinates * self.coord_conversion_coefficient,
        )

        # TODO: ask tblite devs about non-verbose mode?
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")

        try:
            res = calc.singlepoint()
            output = res.dict()
        except:
            if self.display_problematic_xTB_tps:
                sys.stdout = old_stdout  # reset old stdout
                print("#PROBLEMATIC_xTB_TP:", tp)
            output = None

        sys.stdout = old_stdout  # reset old stdout

        return output


class FF_xTB_quantity:
    def __init__(self, quant_name=None, **FF_xTB_res_dict_kwargs):
        self.quant_name = quant_name
        self.res_dict_generator = FF_xTB_res_dict(**FF_xTB_res_dict_kwargs)
        self.res_related_dict = {"res_dict": self.res_dict_generator}
        self.call_counter = 0

    def __call__(self, tp):
        self.call_counter += 1
        res_dict = tp.calc_or_lookup(self.res_related_dict)["res_dict"]
        if res_dict is None:
            return None
        else:
            return self.processed_res_dict(res_dict)

    def processed_res_dict(self, res_dict):
        return res_dict[self.quant_name]


class FF_xTB_HOMO_LUMO_gap(FF_xTB_quantity):
    def processed_res_dict(self, res_dict):
        orb_ens = res_dict["orbital-energies"]
        for orb_id, (orb_en, orb_occ) in enumerate(
            zip(orb_ens, res_dict["orbital-occupations"])
        ):
            if orb_occ < 0.1:
                LUMO_en = orb_en
                HOMO_en = orb_ens[orb_id - 1]
                return LUMO_en - HOMO_en


class FF_xTB_dipole(FF_xTB_quantity):
    def processed_res_dict(self, res_dict):
        dipole_vector = res_dict["dipole"]
        return np.sqrt(np.sum(dipole_vector**2))


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
