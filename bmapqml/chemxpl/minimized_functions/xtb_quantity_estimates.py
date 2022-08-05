# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
import numpy as np
from ..utils import (
    canonical_SMILES_from_tp,
    coord_info_from_tp,
    chemgraph_from_ncharges_coords,
    InvalidAdjMat,
)
from ...utils import read_xyz_lines
from ...data import conversion_coefficient
import os, sys
from tblite.interface import Calculator

from leruli import graph_to_geometry
from leruli.internal import LeruliInternalError

# For quantity estimates with xTB and MMFF coordinates.
class LeruliCoordCalc:
    def __init__(self, time_check=1.0e-2):

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
