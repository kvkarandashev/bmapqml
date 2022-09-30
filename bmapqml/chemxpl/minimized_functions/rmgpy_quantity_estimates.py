# from .morfeus_quantity_estimates import morfeus_coord_info_from_tp
from ..utils import coord_info_from_tp, trajectory_point_to_canonical_rdkit

from rmgpy.rmg.main import Species
from rmgpy.data.solvation import SolvationDatabase
from rmgpy import settings
import os


class RMGSolvation:
    def __init__(self, num_attempts=1, ff_type="MMFF", solvent_label="water"):
        """
        Checks that coordinates of the molecule can be generated via morfeus, and if yes use RMG to estimate the solvation energy.
        """
        self.num_attempts = num_attempts
        self.ff_type = ff_type

        self.coord_info_name = "coord_info"

        self.coord_func_dict = {self.coord_info_name: coord_info_from_tp}
        self.coord_func_kwargs = {
            self.coord_info_name: {
                "num_attempts": self.num_attempts,
                "ff_type": self.ff_type,
            }
        }

        self.solvent_label = solvent_label

        self.solvationDatabase = SolvationDatabase()
        self.solvationDatabase.load(
            os.path.join(settings["database.directory"], "solvation")
        )

        self.solvent_data = self.solvationDatabase.get_solvent_data(self.solvent_label)

    #        solvationDatabase.load(os.path.join(settings['database.directory'], 'solvation'))
    def __call__(self, trajectory_point_in):
        # Sanity check: verify that RdKit can generate coordinates for this chemical graph.
        if self.num_attempts != 0:
            coord_info = trajectory_point_in.calc_or_lookup(
                self.coord_func_dict, kwargs_dict=self.coord_func_kwargs
            )[self.coord_info_name]
            if coord_info["coordinates"] is None:
                return None
            SMILES = coord_info["canon_rdkit_SMILES"]
        else:
            SMILES = trajectory_point_to_canonical_rdkit(
                trajectory_point_in, SMILES_only=True
            )
        # As a byproduct we get the SMILES.
        # Calculate the solvation energy.
        solute_spc = Species().from_smiles(SMILES)
        solute_spc.generate_resonance_structures()
        #        try:
        solute_data = self.solvationDatabase.get_solute_data_from_groups(solute_spc)
        #        except:
        #            return None
        #        try:
        return self.solvationDatabase.calc_g(solute_data, self.solvent_data)
        # except:
        #    return None
