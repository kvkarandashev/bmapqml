import numpy as np
from numpy.linalg import norm
from ..utils import trajectory_point_to_canonical_rdkit
from .. import rdkit_descriptors
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morpheus_coord_info_from_tp,
)
import pdb

try:
    from qml.representations import *
    from qml.kernels import get_local_kernel
except:
    print("local_space_sampling: qml not installed")

try:
    from ase import Atoms
    from dscribe.descriptors import SOAP
except:
    print("local_space_sampling: ase or dscribe not installed")


def gen_soap(crds, chgs):
    # average output
    # https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
    """
    Generate the average SOAP, i.e. the average of the SOAP vectors is
    a global of the molecule.
    """
    average_soap = SOAP(
        rcut=6.0,
        nmax=8,
        lmax=6,
        average="inner",
        species=["H", "C", "N", "O", "F"],
        sparse=False,
    )

    molecule = Atoms(numbers=chgs, positions=crds)
    return average_soap.create(molecule)[0]


def gen_cm(crds, chgs, size=100):
    """
    Generate the coulomb matrix for a set
    of coordinates and their charges.
    """

    return generate_coulomb_matrix(chgs, crds, size=size, sorting="row-norm")


def gen_bob(crds, chgs, size=100):
    """
    Generate the Bag of Bonds representation for a set
    of coordinates and their charges.
    """

    atomtypes = [1, 6, 7, 8, 9]
    asize = {"H": 30, "C": 10, "N": 10, "O": 10, "F": 10}
    return generate_bob(chgs, crds, atomtypes, size=size, asize=asize)


def gen_fchl(crds, chgs, size=100):
    """
    Generate the FCHL representation for a set
    of coordinates and their charges.
    """

    atomtypes = [1, 6, 7, 8, 9]
    return generate_fchl_acsf(chgs, crds, elements=atomtypes, gradients=False, pad=size)


class sample_local_space_3d:
    def __init__(
        self,
        X_init,
        Q_init,
        verbose=False,
        epsilon=1.0,
        sigma=1.0,
        gamma=1,
        repfct=gen_cm,
    ):

        self.repfct = repfct
        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.X_init = X_init
        self.Q_init = Q_init
        self.verbose = verbose
        self.morpheus_output = {"morpheus": morpheus_coord_info_from_tp}
        self.potential = self.flat_parabola_potential

        if (
            self.repfct.__name__ == "gen_cm"
            or self.repfct.__name__ == "gen_bob"
            or self.repfct.__name__ == "gen_soap"
        ):
            self.distfct = self.euclidean_distance

        elif self.repfct.__name__ == "gen_fchl":
            self.distfct = self.fchl_distance

    def euclidean_distance(self, coords, charges):
        """
        Compute the euclidean distance between the test
        point and the initial point.
        """
        X_test = self.repfct(coords, charges)
        return norm(X_test - self.X_init)

    def fchl_distance(self, coords, charges, sigma=1.0):
        """
        Compute the FCHL distance between the test
        point and the initial point.
        """

        X_test = self.repfct(coords, charges)
        k = get_local_kernel(
            np.array([X_test]), np.array([self.X_init]), [charges], [self.Q_init], sigma
        )[0][0]
        return 500 - k

    def flat_parabola_potential(self, d):

        """
        Flat parabola potential. Allows sampling within a distance basin
        interval of I in [gamma, sigma]. epsilon determines depth of minima
        The values as well as slope must be adjusted depending on the representation.
        The potential is given by:
        """

        if d < self.gamma:
            return 0.05 * (d - self.gamma) ** 2 - self.epsilon
        if self.gamma <= d <= self.sigma:
            return -self.epsilon
        if d > self.sigma:
            return 0.05 * (d - self.sigma) ** 2 - self.epsilon

    def __call__(self, trajectory_point_in):
        """
        Compute the potential energy of a trajectory point.
        """

        try:
            output = trajectory_point_in.calc_or_lookup(self.morpheus_output)[
                "morpheus"
            ]
        except:
            print("Error in 3d conformer sampling")
            return None

        coords = output["coordinates"]
        charges = output["nuclear_charges"]
        SMILES = output["canon_rdkit_SMILES"]

        try:
            distance = self.distfct(coords, charges)
            V = self.potential(distance)
            return V

        except:
            if coords is None:
                print(f"{SMILES} {distance} {V}")
                print("Error in 3d conformer sampling")
            else:
                print("sth else happened")
            return None


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
        nbits=4096,
    ):

        self.fp_type = None or fp_type
        self.epsilon = epsilon
        self.check_ring = None or check_ring
        self.sigma = sigma
        self.gamma = gamma
        self.X_init = X_init
        self.pot_type = pot_type
        self.verbose = verbose
        self.nbits = nbits
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
        Flat parabola potential. Allows sampling within a distance basin
        interval of I in [gamma, sigma]. epsilon determines depth of minima
        and is typically set to epsilon = 5. The potential is given by:
        """

        if d < self.gamma:
            return 0.05 * (d - self.gamma) ** 2 - self.epsilon
        if self.gamma <= d <= self.sigma:
            return -self.epsilon
        if d > self.sigma:
            return 0.05 * (d - self.sigma) ** 2 - self.epsilon

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

        X_test = rdkit_descriptors.extended_get_single_FP(
            canon_SMILES, self.fp_type, nBits=self.nbits
        )

        d = norm(X_test - self.X_init)
        V = self.potential(d)

        return V

    def evaluate_point(self, trajectory_point_in):
        """
        Evaluate the function on a list of trajectory points
        """

        _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = rdkit_descriptors.extended_get_single_FP(
            canon_SMILES, self.fp_type, nBits=self.nbits
        )
        d = norm(X_test - self.X_init)
        V = self.potential(d)

        return V, d

    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points
        """

        values = []
        for trajectory_point in trajectory_points:
            values.append(self.evaluate_point(trajectory_point))

        return np.array(values)


class local_lipinski:
    # if self.check_ring:
    #    try:
    #        ring_error = self.get_largest_ring_size(canon_SMILES)
    #    except:
    #        ring_error = 0
    #
    #    V += ring_error
    # if self.verbose:
    # print("SMILE:", canon_SMILES, d, V)
    def __init__(
        self,
        X_init,
        verbose=False,
        fp_type=None,
        epsilon=1.0,
        sigma=1.0,
        gamma=1,
    ):

        self.fp_type = None or fp_type
        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.X_init = X_init
        self.verbose = verbose
        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

        self.potential = self.flat_parabola_potential

    def flat_parabola_potential(self, d):
        """
        Flat parabola potential. The potential is given by:
        """

        if d < self.gamma:
            return 0.05 * (d - self.gamma) ** 2 - self.epsilon
        if self.gamma <= d <= self.sigma:
            return -self.epsilon
        if d > self.sigma:
            return 0.05 * (d - self.sigma) ** 2 - self.epsilon

    def log_partition_coefficient(smiles):
        """
        Returns the octanol-water partition coefficient given a molecule SMILES
        string
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            print("Error in canonical SMILES, therefore skipping")
            return None

        return Crippen.MolLogP(mol)

    def lipinski_trial(self, smiles):

        """
        Lipinski's rules are:
        Hydrogen bond donors <= 5
        Hydrogen bond acceptors <= 5
        Molecular weight < 500 daltons
        logP < 5
        """

        mol = Chem.MolFromSmiles(smiles)
        num_hdonors = Lipinski.NumHDonors(mol)
        num_hacceptors = Lipinski.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Crippen.MolLogP(mol)
        mol_rot = Lipinski.NumRotatableBonds(mol)
        mol_ring = Lipinski.RingCount(mol)

        print(num_hdonors, num_hacceptors, mol_weight, mol_logp)
        if (
            num_hdonors <= 5
            and num_hacceptors <= 5
            and mol_weight < 500
            and mol_logp < 5
            and mol_rot <= 5
            and mol_ring > 0
        ):
            return True
        else:
            return False

    def __call__(self, trajectory_point_in):

        try:
            _, _, _, canon_SMILES = trajectory_point_in.calc_or_lookup(
                self.canonical_rdkit_output
            )["canonical_rdkit"]
        except:
            print("Error in canonical SMILES, therefore skipping")
            return None

        if self.lipinski_trial(canon_SMILES):
            pass
        else:
            print("skipping because of Lipinski")
            return None

        X_test = rdkit_descriptors.extended_get_single_FP(canon_SMILES, self.fp_type)

        d = norm(X_test - self.X_init)
        V = self.potential(d)

        if self.verbose:
            print("SMILE:", canon_SMILES, d, V)

        return V


class find_match:
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
