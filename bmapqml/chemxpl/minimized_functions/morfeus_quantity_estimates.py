from morfeus.conformer import conformers_from_rdkit
from ..utils import (
    chemgraph_to_canonical_rdkit,
    FFInconsistent,
    InvalidAdjMat,
    chemgraph_from_ncharges_coords,
)
from ...utils import NUCLEAR_CHARGE
from .xtb_quantity_estimates import FF_xTB_res_dict, FF_xTB_HOMO_LUMO_gap, FF_xTB_dipole
import numpy as np


def morpheus_coord_info_from_tp(
    tp, coords=None, num_attempts=1, ff_type="MMFF94", **dummy_kwargs
):
    """
    Coordinates corresponding to a TrajectoryPoint object
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "nuclear_charges": None, "canon_rdkit_SMILES": None}
    cg = tp.egc.chemgraph
    canon_rdkit_mol, _, _, canon_rdkit_SMILES = chemgraph_to_canonical_rdkit(cg)
    output["canon_rdkit_SMILES"] = canon_rdkit_SMILES
    try:
        conformers = conformers_from_rdkit(
            canon_rdkit_mol, n_confs=num_attempts, optimize=ff_type
        )
    except ValueError:
        return output

    nuclear_charges = np.array([NUCLEAR_CHARGE[el] for el in conformers[0]])
    coordinates = conformers[1][np.argmin(conformers[2])]

    try:
        coord_based_cg = chemgraph_from_ncharges_coords(nuclear_charges, coordinates)
    except InvalidAdjMat:
        return output
    if coord_based_cg != cg:
        return output

    output["coordinates"] = coordinates
    output["nuclear_charges"] = nuclear_charges

    return output


xTB_quant_morpheus_kwargs = {
    "coord_calculation_type": "morfeus",
    "coord_info_from_tp_func": morpheus_coord_info_from_tp,
}


def morfeus_FF_xTB_HOMO_LUMO_gap(**kwargs):
    return FF_xTB_HOMO_LUMO_gap(**xTB_quant_morpheus_kwargs, **kwargs)


def morfeus_FF_xTB_dipole(**kwargs):
    return FF_xTB_dipole(**xTB_quant_morpheus_kwargs, **kwargs)
