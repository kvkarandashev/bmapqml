# MIT License
#
# Copyright (c) 2021-2022 Konstantin Karandashev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Miscellaneous functions and classes used throughout the code.
import pickle, subprocess

from matplotlib.pyplot import fignum_exists
from .data import NUCLEAR_CHARGE
import numpy as np
import pdb

def canonical_atomtype(atomtype):
    return atomtype[0].upper()+atomtype[1:].lower()

def dump2pkl(obj, filename):
    output_file = open(filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()

def loadpkl(filename):
    input_file = open(filename, "rb")
    obj=pickle.load(input_file)
    input_file.close()
    return obj
    
def mktmp(directory=False):
    extra_args=()
    if directory:
        extra_args=("-d", *extra_args)
    return subprocess.check_output(["mktemp", *extra_args, "-p", "."], text=True).rstrip("\n")

def mktmpdir():
    return mktmp(True)
   
def mkdir(dir_name):
    subprocess.run(["mkdir", "-p", dir_name])
 
def rmdir(dirname):
    subprocess.run(["rm", "-Rf", dirname])

#   XYZ processing.
def check_byte(byte_or_str):
    if isinstance(byte_or_str, str):
        return byte_or_str
    else:
        return byte_or_str.decode('utf-8')


def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, compound.atomtypes, xyz_file_name)

def write_xyz_file(coordinates, elements, xyz_file_name):
    xyz_file=open(xyz_file_name, 'w')
    xyz_file.write(str(len(coordinates))+'\n\n')
    for atom_coords, element in zip(coordinates, elements):
        xyz_file.write(element+" "+' '.join([str(atom_coord) for atom_coord in atom_coords])+'\n')
    xyz_file.close()

def read_xyz_file(xyz_input, additional_attributes=["charge"]):
    atomic_symbols = []
    add_attr_dict={}
    for add_attr in additional_attributes:
        add_attr_dict={add_attr : None, **add_attr_dict}

    try:
        lines=[check_byte(l) for l in xyz_input.readlines()]
    except AttributeError:
        with open(xyz_input, "r") as input_file:
            lines=input_file.readlines()
    num_atoms=int(lines[0])
    xyz_coordinates=np.zeros((num_atoms, 3))
    nuclear_charges=np.zeros((num_atoms,), dtype=int)

    lsplit=lines[1].split()
    for l in lsplit:
        for add_attr in additional_attributes:
            add_attr_eq=add_attr+"="
            if add_attr_eq == l[:len(add_attr_eq)]:
                add_attr_dict[add_attr]=int(l.split("=")[1])

    for atom_id, atom_line in enumerate(lines[2:num_atoms+2]):
        lsplit=atom_line.split()
        atomic_symbol = lsplit[0]
        atomic_symbols.append(atomic_symbol)
        nuclear_charges[atom_id]=NUCLEAR_CHARGE[canonical_atomtype(atomic_symbol)]
        for i in range(3):
            xyz_coordinates[atom_id, i]=float(lsplit[i+1])

    return nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict

def write2file(string, file_name):
    file_output=open(file_name, 'w')
    print(string, file=file_output)
    file_output.close()

class OptionUnavailableError(Exception):
    pass


def merged_representation_array(total_compound_array):
    return [np.array([total_compound_array[comp_id].representation for comp_id in range(*index_tuple)])]

def where2slice(indices_to_ignore):
    return np.where(np.logical_not(indices_to_ignore))[0]


def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(np.logical_not(cur_ignore_indices))]=0.0

#   A dumb way to run commands without regards for spaces inside them when subprocess.run offers no viable workarounds..
def execute_string(string):
    script_name=mktmp()
    subprocess.run(["chmod", "+x", script_name])
    script_output=open(script_name, 'w')
    script_output.write("#!/bin/bash\n"+string)
    script_output.close()
    subprocess.run(["chmod", "+x", script_name])
    subprocess.run(["./"+script_name])
    rmdir(script_name)





def trajectory_point_to_canonical_rdkit(tp_in):
    from bmapqml.chemxpl.utils import chemgraph_to_canonical_rdkit   
    return chemgraph_to_canonical_rdkit(tp_in.egc.chemgraph)

class analyze_random_walk:

    """
    Class for analyzing a random walk after the simulation.
    Visualize chemical space and convex hull of the walk.
    """

    def __init__(self, histogram, saved_candidates, model=None, verbose=False):
        import numpy as np
        
        """
        histogram : list of all unique encountered points in the random walk.
        saved_candidates : list of all saved best candidates in the random walk.
        minimize_fct : if desired reevaluate all unique molecules using this function
        """

        import pickle
        self.tps     = pickle.load(open(histogram, "rb"))
        self.histogram= np.array(self.convert_to_smiles(pickle.load(open(histogram, "rb"))))
        self.saved_candidates=pickle.load(open(saved_candidates, "rb"))
        self.saved_candidates_tps, self.saved_candidates_func_val = [],[]
        self.model = None or model


        for mol in self.saved_candidates:
            self.saved_candidates_tps.append(mol.tp)
            self.saved_candidates_func_val.append(mol.func_val)

    
        del(self.saved_candidates)
        self.saved_candidates = self.convert_to_smiles(self.saved_candidates_tps)
        self.saved_candidates_func_val = np.array(self.saved_candidates_func_val)
        
        if verbose:
            print("Number of unique points in the random walk:", len(self.tps))
    
    def convert_to_smiles(self, mols):

        """
        Convert the list of molecules to SMILES strings.
        """

        import rdkit
        from rdkit import Chem

        """
        tp_list: list of molecules as trajectory points
        smiles_mol: list of rdkit molecules
        """
        smiles_mol = []
        for tp in mols:
            
            rdkit_obj = trajectory_point_to_canonical_rdkit(tp)
            smiles_mol.append(rdkit_obj[3])

        return smiles_mol

    def evaluate_histogram(self, save_histogram=False):
        import numpy as np
        """
        Compute values of a function evaluated on all unique smiles in the random walk.
        """

        self.values = self.model.evaluate_trajectory(self.tps)

        if save_histogram:
            np.savez_compressed("QM9_histogram_values", values=self.values)

        return self.values

    def compute_pareto_front(self):
        
        """
        values: array of function values
        Returns values of points in the pareto front.
        pareto_front as well as the indices
        """
        
        try:
            two_dim = self.values[:,:2]
        except:
            self.values = np.load("QM9_histogram_values.npz")["values"]
            two_dim = self.values[:,:2]

        """    
        else:
            self.evaluate_histogram(save_histogram=False)
            two_dim = self.values[:,:2]
        finally:
            print("Could not load property values")
        """

        Xs, Ys = two_dim[:,0], two_dim[:,1]
        maxY = False
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
        pareto_front = [sorted_list[0]]
        
        for pair in sorted_list[1:]:
            if maxY:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)

        inds = []   
        for pair in pareto_front:
            if pair[0] in two_dim[:,0] and pair[1] in two_dim[:,1]:
                inds.append( int(np.where(two_dim[:,0]==pair[0]) and np.where(two_dim[:,1]==pair[1])[0][0]))
        
        inds =np.array(inds)
        inds  = inds.astype('int')


        self.pareto_mols = self.histogram[inds]

        return np.array(pareto_front), np.int_(inds), self.pareto_mols