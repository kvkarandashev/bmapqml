import os
import subprocess

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from ..utils import vector_to_square


class DGSOL:
    """
    Wrapper class for the Distance Geometry Solver (DGSOL)

    Embeds points in cartesian space given a distance boundary.

    To read more about DGSOl visit: https://www.mcs.anl.gov/~more/dgsol/
    """

    def __init__(self, distances, nuclear_charges, vectorized_input=True):
        """

        Parameters
        ----------
        distances: np.array
            Either a symmetric (n, n) distance matrix or its vectorized form.
        nuclear_charges: np.array, shape n
            Nucelear charges of the system
        vectorized_input: bool (default=True)
            Whether the distance matrix is in its vectorized form or not.
            If True, converts distance matrix to its symmetric form.
        """
        self.nuclear_charges = nuclear_charges
        self.distances = vector_to_square(distances) if vectorized_input else distances
        self.coords = None
        self.c_errors = None

    def gen_cerror_overview(self):
        """
        Prints overview of DGSOl reconstruction errors
        """
        print("Error Type, Min, Mean, Max")
        print(
            f"minError: {np.min(self.c_errors[:, 1])}, {np.mean(self.c_errors[:, 1])}, {np.max(self.c_errors[:, 1])}"
        )
        print(
            f"avgError: {np.min(self.c_errors[:, 2])}, {np.mean(self.c_errors[:, 2])}, {np.max(self.c_errors[:, 2])}"
        )
        print(
            f"maxError: {np.min(self.c_errors[:, 2])}, {np.mean(self.c_errors[:, 3])}, {np.max(self.c_errors[:, 3])}"
        )

    def to_scientific_notation(self, number):
        """
        Converts numbers to DGSOL notation.

        Parameters
        ----------
        number: float

        Returns
        -------
        Number in DGSOL notation, e.g. 1e10
        """
        a, b = "{:.17E}".format(number).split("E")
        num = "{:.12f}E{:+03d}".format(float(a) / 10, int(b) + 1)
        return num[1:]

    def write_dgsol_input(self, distances, outpath):
        """
        Input file writer for DGSOL.
        Basically writes 4 columns such as
        Atom_i   Atom_j  lower_bound       upper_bound
        1         2   .139169904722E+01   .139169904722E+01
        1         3   .237179033727E+01   .237179033727E+01
        1         4   .331764447534E+01   .331764447534E+01
        1         5   .200997900174E+01   .200997900174E+01

        Parameters
        ----------
        distances: np.array
            Vectorized distance matrix.
        outpath: str
            Directory to save input file

        """
        n, m = np.triu_indices(distances.shape[1], k=1)
        with open(f"{outpath}/dgsol.input", "w") as outfile:
            for i, j in zip(n, m):
                outfile.write(
                    f"{i + 1:9.0f}{j + 1:10.0f}   {self.to_scientific_notation(distances[i, j])}   "
                    f"{self.to_scientific_notation(distances[i, j])}\n"
                )

    def parse_dgsol_coords(self, path, n_solutions, n_atoms):
        """
        Parser for DGSOl output file.
        Reads all found solutions and filters coordinates.

        Parameters
        ----------
        path: str
            Path to dgsol.output file.
        n_solutions: int
            Number of dgsol solutions.
        n_atoms: int
            Number of atoms.

        Returns
        -------
        coords: np.array, shape (n, 3)
            Coordinates of the system.

        """
        with open(f"{path}/dgsol.output") as outfile:
            lines = outfile.readlines()

        coords = []
        for line in lines:
            if not line.startswith("\n") and len(line) > 30:
                coords.append([float(n) for n in line.split()])
        coords = np.array(coords).reshape((n_solutions, n_atoms, 3))
        return coords

    def check_coords(self, coords):
        idx = np.where(coords >= 1e4)
        return np.unique(idx[0])

    def solve_distance_geometry(self, outpath, n_solutions=10, n_cpus=1):
        """
        Interface to solve distance geometry problem.
        Writes input for DGSOL, run's DGSOL and parses coordinates.

        Parameters
        ----------
        outpath: str
            Output directory to write input files and run DGSOL.
        n_solutions: int (default=10)
            Number of solutions to compute with DGSOL.
        n_cpus: int (default=1)
            Number of cpus to use.

        """
        construction_errors = []
        mol_coordinates = []

        jobs = [
            (outpath, n_solutions, idx) for idx in np.arange(self.distances.shape[0])
        ]
        with mp.Pool(processes=n_cpus) as pool:
            for res in tqdm(
                pool.imap(self.run_reconstruct, jobs), total=len(jobs), desc="DGSOL"
            ):
                construction_errors.append(res[0])
                mol_coordinates.append(res[1])
        self.coords = mol_coordinates
        self.c_errors = construction_errors

    def run_reconstruct(self, data):
        outpath, n_solutions, idx = data
        n_atoms = len(self.nuclear_charges[idx])
        out = f"{outpath}/{idx:06}"
        os.makedirs(out, exist_ok=True)
        self.write_dgsol_input(distances=self.distances[idx], outpath=out)
        self.run_dgsol(out, n_solutions=n_solutions)
        errors = self.parse_dgsol_errors(out)
        coords = self.parse_dgsol_coords(out, n_solutions, n_atoms=n_atoms)
        bad_ids = self.check_coords(coords)
        good_ids = np.setdiff1d(np.arange(n_solutions), bad_ids, assume_unique=True)
        if good_ids.size == 0:
            lowest_errors_idx = np.nanargmin(errors[:, 2])
            return errors[lowest_errors_idx], np.zeros((n_atoms, 3))
        else:
            errors = errors[good_ids]
            coords = coords[good_ids]
            lowest_errors_idx = np.nanargmin(errors[:, 2])
            return errors[lowest_errors_idx], coords[lowest_errors_idx]

    def run_dgsol(self, outpath, n_solutions=10):
        """
        Interface to submit DGSOL as a subprocess.

        Parameters
        ----------
        outpath: str
            Output directory to write input files and run DGSOL.
        n_solutions: int (default=10)
            Number of solutions to compute with DGSOL.

        """
        cmd = f"dgsol -s{n_solutions} {outpath}/dgsol.input {outpath}/dgsol.output {outpath}/dgsol.summary"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error is not None:
            raise UserWarning(f"{outpath} produced the following error: {error}")

    def parse_dgsol_errors(self, outpath):
        """
        Parses DGSOL Errors.

        There are 4 types of errors in the dgsol output:

        f_err         The value of the merit function
        derr_min      The smallest error in the distances
        derr_avg      The average error in the distances
        derr_max      The largest error in the distances

        Parameters
        ----------
        outpath: str
            Output directory that contains dgsol.summary

        Returns
        -------
        dgsol_erros: np.array
            Contains DGSOL errors, shape(4)

        """
        with open(f"{outpath}/dgsol.summary", "r") as input:
            lines = input.readlines()

        errors = []
        # skip the header lines
        for line in lines[5:]:
            errors.append(
                line.split()[2:]
            )  # the first two entries are n_atoms and n_distances
        return np.array(errors).astype("float32")
