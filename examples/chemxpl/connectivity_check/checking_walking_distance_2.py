import sys

# The chemical space of choice is the same as used here:
# https://github.com/kvkarandashev/molopt/blob/main/examples/chemxpl/bias_potential_checks/MC_graph_enumeration.py
# It is defined as chemical space of up to 5 heavy atoms (C, P, or O), with O not protonated, and P-P and O-O covalent bonds forbidden.
chemspace_def = {
    "max_fragment_num": 1,  # maximal number of fragments a molecule is allowed to break into
    "nhatoms_range": [1, 5],  # numbers of heavy atoms allowed in the system
    "final_nhatoms_range": [
        1,
        5,
    ],  # numbers of heavy atoms of "principle interest" (simulation is biased towards that if bound_enfocing_coeff in RandomWalk is not None)
    "possible_elements": ["C", "P", "O"],  # allowed elements
    "forbidden_bonds": [(15, 15), (8, 8)],
    "not_protonated": [8],
}
# It is relatively small (11507 compounds found), showcases the way we implemented constraints
# against protonation and formation of unstable covalent bonds, and code utilizes all elementary mutations M1-M6 to traverse it.
bias_coeff = 0.0  # in the paper we check values 0.0, 0.2, and 0.4


sys.path.append("../../../")
sys.path.append("../../../../Graph2Structure")
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.random_walk import (
    RandomWalk,
    TrajectoryPoint,
    minimized_change_list,
    full_change_list,
)
from bmapqml.chemxpl.utils import SMILES_to_egc
import random
import numpy as np


def EGC_walking_distance(
    egc1: ExtGraphCompound,
    egc2: ExtGraphCompound,
    rw: RandomWalk,
    max_num_MC_steps: int = 50000,
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
    tp_final = TrajectoryPoint(egc=egc2)

    rw.init_cur_tps([egc1])
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
    max_num_MC_steps = 500000

    elementary_change_list = full_change_list  # use all M1-M6 mutations
    #    elementary_change_list=minimized_change_list # only use M1-M4 mutations

    randomized_change_params = {
        **chemspace_def,
        "change_prob_dict": elementary_change_list,
        "bond_order_changes": [-1, 1],  # values by which bond orders can be changed
    }

    seed = 42
    # The code codes two different number generators at different point.
    random.seed(seed)
    np.random.seed(seed)

    SMILES1 = "CCC"
    SMILES2 = "C1PCOC1"
    print("STARTING_MOLECULE", SMILES1)
    print("TARGET_MOLECULE", SMILES2)

    num_attempts = 100

    print("# steps needed, molecules seen")
    for attempt_counter in range(num_attempts):
        rw = RandomWalk(
            bias_coeff=None,
            randomized_change_params=randomized_change_params,
            keep_histogram=True,
            num_replicas=1,
        )
        dist = SMILES_walking_distance(
            SMILES1, SMILES2, rw, max_num_MC_steps=max_num_MC_steps
        )
        nmolecules = len(rw.histogram)
        print(dist, nmolecules)
