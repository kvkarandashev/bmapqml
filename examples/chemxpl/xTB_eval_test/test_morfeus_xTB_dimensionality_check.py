from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
)
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.dataset_processing.qm9_format_specs import Quantity
from bmapqml.data import conversion_coefficient
import os

SMILES_xyzs = [
    ("C", "dsgdb9nsd_000001.xyz"),
    ("CC#N", "dsgdb9nsd_000010.xyz"),
    ("CCO", "dsgdb9nsd_000014.xyz"),
]

for SMILES, xyz in SMILES_xyzs:

    print("SMILES:", SMILES)

    QM9_xyz = os.environ["DATA"] + "/QM9_formatted/" + xyz

    QM9_dipole = Quantity("Dipole moment").extract_xyz(QM9_xyz)
    QM9_gap = Quantity("HOMO-LUMO gap").extract_xyz(QM9_xyz)

    print("QM9 reference")
    print("Dipole (a.u.) # HOMO-LUMO gap (a.u.)")
    print(QM9_dipole * conversion_coefficient["Debye_au"], QM9_gap)

    print("Properties of ")

    tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))

    print("Morfeus + xTB results")
    print(
        morfeus_FF_xTB_code_quants(
            tp, num_attempts=4, quantities=["HOMO_LUMO_gap", "dipole"]
        )
    )
