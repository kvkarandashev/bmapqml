# Contains constraints that can be used with ConstrainedQuant from quantity_estimates.py
from ..random_walk import tidy_forbidden_bonds
from ..modify import no_forbidden_bonds


class NoProtonation:
    def __init__(self, restricted_ncharges=[]):
        """
        Forbids TrajectoryPoint objects that correspond to molecules containing hydrogens covalently connected to atoms with nuclear charge present in restricted_ncharges.
        """
        self.call_counter = 0
        self.restricted_ncharges = restricted_ncharges

    def __call__(self, trajectory_point_in):
        self.call_counter += 1
        cg = trajectory_point_in.egc.chemgraph
        for hatom in cg.hatoms:
            if (hatom.nhydrogens != 0) and (hatom.ncharge in self.restricted_ncharges):
                return False
        return True


class NoForbiddenBonds:
    def __init__(self, forbidden_bonds=[]):
        """
        Forbids TrajectoryPoint objects that correspond to molecules containing covalent bonds between pairs of atoms with nuclear charges listed in forbidden bonds.
        """
        self.forbidden_bonds = tidy_forbidden_bonds(forbidden_bonds)
        self.call_counter = 0

    def __call__(self, trajectory_point_in):
        self.call_counter += 1
        return no_forbidden_bonds(
            trajectory_point_in.egc, forbidden_bonds=self.forbidden_bonds
        )
