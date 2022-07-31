# If we explore diatomic molecule graph, this function will create chemgraph analogue of a double-well potential.
import numpy as np
from numpy.linalg import norm
from ..utils import trajectory_point_to_canonical_rdkit
from joblib import Parallel, delayed
from .. import rdkit_descriptors
from rdkit import Chem
from tqdm import tqdm


class sample_local_space:

    """
    Sample local chemical space of the inital compound
    with input representation X_init. Radial symmetric
    LJ12-6 potential is used. Epsilon and sigma parameters
    can be changed to adjust the potential and how different
    the target molecule can be from the initial one.
    """

    def __init__(
        self,
        X_init,
        verbose=False,
        check_ring=False,
        pot_type="harmonic",
        fp_type=None,
        epsilon=1.0,
        sigma=1.0,
        gamma=1,
    ):

        self.fp_type = None or fp_type
        self.epsilon = epsilon
        self.check_ring = None or check_ring
        self.sigma = sigma
        self.gamma = gamma
        self.X_init = X_init
        self.pot_type = pot_type
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

        self.potential = None
        if self.pot_type == "lj":
            self.potential = self.lennard_jones_potential
        elif self.pot_type == "harmonic":
            self.potential = self.harmonic_potential
        elif self.pot_type == "buckingham":
            self.potential = self.buckingham_potential
        elif self.pot_type == "exponential":
            self.potential = self.exponential_potential
        elif self.pot_type == "sharp_parabola":
            self.potential = self.sharp_parabola_potential
        elif self.pot_type == "flat_parabola":
            self.potential = self.flat_parabola_potential

    def get_largest_ring_size(self, SMILES):


        """ 
        Returns the size of the largest ring in the molecule.
        If ring too large (>7) reject that move and return large energy
        """

        m = Chem.MolFromSmiles(SMILES)
        ri = m.GetRingInfo()
        all_rings = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            all_rings.append(np.array(list(ringAts)))

        all_rings = np.array(all_rings)
        max_size = max(map(len, all_rings))

        if max_size < 7:
            return 0
        else:
            return None

    def lennard_jones_potential(self, d):
        """
        Lennard-Jones potential, d is the distance between the two points
        in chemical space. The potential is given by:
        """
        return 4 * self.epsilon * ((self.sigma / d) ** 12 - (self.sigma / d) ** 6)

    def harmonic_potential(self, d):
        """
        Flat-bottomed harmonic potential. within the range of 0 to sigma
        the potential is flat and has value epsilon. Outsite it is quadratic.
        This avoids the problem of the potential being infinite at zero such that
        also molecule close to or equal to d = 0 can be sampled
        """

        if d <= self.sigma:
            return -self.epsilon
        else:
            return (d - self.sigma) ** 2 - self.epsilon

    def sharp_parabola_potential(self, d):
        """
        Sharp parabola potential. The potential is given by:
        """
        return 20 * (d - self.sigma) ** 2 - self.epsilon

    def flat_parabola_potential(self, d):
        """
        Flat parabola potential. The potential is given by:
        """

        if d < self.gamma:
            return (d - self.gamma) ** 2 - self.epsilon
        if self.gamma <= d <= self.sigma:
            return -self.epsilon
        if d > self.sigma:
            return (d - self.sigma) ** 2 - self.epsilon

    def buckingham_potential(self, d):
        """
        Returns the buckingham potential for a given distance.
        It is a combination of an exponential and a one over 6 power
        using the parameters epsilon and sigma and gamma.
        """

        return self.epsilon * np.exp(-self.sigma * d) - self.gamma * (1 / d**6)

    def exponential_potential(self, d):
        """
        Returns the exponential potential for a given distance.
        """

        return self.epsilon * np.exp(-self.sigma * d)

    def __call__(self, trajectory_point_in):


        try:
            _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
                self.canonical_rdkit_output
            )["canonical_rdkit"]
        except:
            print("Error in canonical SMILES, therefore skipping")
            return None

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, self.fp_type)

        d = norm(X_test - self.X_init)
        V = self.potential(d)

        if self.check_ring:
            try:
                ring_error = self.get_largest_ring_size(canon_SMILES)
            except:
                ring_error = 0

            V += ring_error

        if self.verbose:
            print("SMILE:", canon_SMILES, d, V)

        return V

    def evaluate_point(self, trajectory_point_in):

        """
        Evaluate the function on a list of trajectory points
        """

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, self.fp_type)
        d = norm(X_test - self.X_init)
        V = self.potential(d)

        return V, d

    def evaluate_trajectory(self, trajectory_points):
        

        """
        Evaluate the function on a list of trajectory points
        """

        # Aparently this is the fastest way to do this:
        values = []
        for trajectory_point in tqdm(trajectory_points):
            values.append(self.evaluate_point(trajectory_point))
        # Parallel(n_jobs=1)(delayed(self.evaluate_point)(tp_in) for tp_in in trajectory_points)
        return np.array(values)



class find_match:

    """
    Guide Random walk towards the target the representation corresponding to the target
    molecule.
    """

    def __init__(self, X_target, verbose=False):

        self.X_target = X_target
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

    def __call__(self, trajectory_point_in):

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, "both")
        d = norm(X_test - self.X_target)

        if self.verbose:
            print("SMILE:", canon_SMILES, "Prediction: ", d)

            return np.exp(d)