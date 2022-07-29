# Contains toy problems for code testing.
from ..random_walk import TrajectoryPoint


class Diatomic_barrier:
    def __init__(self, possible_nuclear_charges: list):
        """
        Toy problem potential: If you can run a Monte Carlo simulation in chemical space of diatomic molecules with two elements available this minimization function will
        create one global minimum for A-A, one local minimum for B-B, and A-B as a maximum that acts as a transition state for single-replica MC moves.
        possible_nuclear_charges : nuclear charges considered
        """
        self.larger_nuclear_charge = max(possible_nuclear_charges)
        # Mainly used for testing purposes.
        self.call_counter = 0

    def __call__(self, trajectory_point_in: TrajectoryPoint) -> float:
        self.call_counter += 1
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
    def __init__(self, possible_nuclear_charges_input: list):
        """
        Toy problem potential for minimizing nuclear charges of heavy atoms of molecules.
        """
        possible_nuclear_charges = sorted(possible_nuclear_charges_input)
        self.order_dict = {}
        for i, ncharge in enumerate(possible_nuclear_charges):
            self.order_dict[ncharge] = i

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        return sum(
            self.order_dict[ha.ncharge]
            for ha in trajectory_point_in.egc.chemgraph.hatoms
        )


class ZeroFunc:
    """
    Unbiased chemical space exploration.
    """

    def __init__(self):
        self.call_counter = 0

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return 0.0


class ChargeSum(ZeroFunc):
    """
    Toy problem potential for minimizing sum of nuclear charges.
    """

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return sum(ha.ncharge for ha in trajectory_point_in.egc.chemgraph.hatoms)
