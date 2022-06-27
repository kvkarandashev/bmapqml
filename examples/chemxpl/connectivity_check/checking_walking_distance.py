import sys

sys.path.append("../../../")
sys.path.append("../../../../Graph2Structure")
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.random_walk import RandomWalk, TrajectoryPoint
from bmapqml.chemxpl.utils import SMILES_to_egc
import random
import math
from bmapqml.chemxpl.modify import (
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
)


def EGC_walking_distance(
    egc1: ExtGraphCompound,
    egc2: ExtGraphCompound,
    rw: RandomWalk,
    max_num_MC_steps: int = 10_000,
) -> int:
    """The function for finding the distance between to EGC files.

    Parameters
    ----------
    egc1 : ExtGraphCompound
        Starting molecule
    egc2 : ExtGraphCompound
        Target molecule
    rw : RandomWalk
        Instance for given random walk
    max_num_MC_steps : int, optional
        Number of steps to be taken at most, by default 10000

    Returns
    -------
    int, None
        Number of steps to reach the target, None if target is not reached.
    """

    rw.clear_histogram_visit_data()
    tp_start = TrajectoryPoint(egc=egc1)
    tp_final = TrajectoryPoint(egc=egc2)

    rw.cur_tps = [tp_start]
    for MC_step in range(max_num_MC_steps):
        if tp_final in rw.cur_tps:
            return MC_step
        rw.MC_step_all()
    return None


def SMILES_walking_distance(SMILES1: str, SMILES2: str, *args, **kwargs) -> int:
    """Distance, but we start from SMILES.

    Parameters
    ----------
    SMILES1 : str
        SMILES string of starting molecule
    SMILES2 : str
        SMILES string of target molecule

    Returns
    -------
    int
        Number of steps taken.
    """

    egc1 = SMILES_to_egc(SMILES1)
    egc2 = SMILES_to_egc(SMILES2)

    return EGC_walking_distance(egc1, egc2, *args, **kwargs)


if __name__ == "__main__":
    # Parameters of the random walk.
    allowed_elements = ["C", "O", "N", "F", "P"]
    max_num_MC_steps = 10_000
    bound_steepness = math.log(2.0)

    # Here you define the probabilities of taking each of the individual smaller moves:
    # If we define it as a list the probabilities are equal:
    change_prob_dict = [
        add_heavy_atom_chain,
        remove_heavy_atom,
        replace_heavy_atom,
        change_bond_order,
        change_valence,
    ]

    randomized_change_params = {
        "max_fragment_num": 1,
        "nhatoms_range": [1, 3],
        "final_nhatoms_range": [2, 3],
        "possible_elements": allowed_elements,
        "bond_order_changes": [-1, 1],
        "forbidden_bonds": None,
        "change_prob_dict": change_prob_dict,
    }

    random.seed(42)

    SMILES1 = "C"
    SMILES2 = "O"
    print("STARTING_MOLECULE", SMILES1)
    print("TARGET_MOLECULE", SMILES2)

    num_attempts = 100

    print("# steps needed, molecules seen")
    for attempt_counter in range(num_attempts):
        rw = RandomWalk(
            bias_coeff=None,
            randomized_change_params=randomized_change_params,
            bound_enforcing_coeff=bound_steepness,
            keep_histogram=True,
            num_replicas=1,
        )
        dist = SMILES_walking_distance(
            SMILES1, SMILES2, rw, max_num_MC_steps=max_num_MC_steps
        )
        nmolecules = len(rw.histogram)
        print(dist, nmolecules)
