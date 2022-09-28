# Contains constraints that can be used with ConstrainedQuant from quantity_estimates.py


class NoProtonation:
    def __init__(self, restricted_ncharges=[]):
        """
        Forbids TrajectoryPoint objects that correspond to molecules containing hydrogens covalently connected to atoms with nuclear charge present in restricted_ncharges.
        """
        self.call_counter = 0
        self.restricted_ncharges = restricted_ncharges

    def __call__(self, trajectory_point_in):
        self.call_counter = 0
        cg = trajectory_point_in.egc.chemgraph
        for hatom in cg.hatoms:
            if (hatom.nhydrogens != 0) and (hatom.ncharge in self.restricted_ncharges):
                return False
        return True
