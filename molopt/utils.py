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
from .data import NUCLEAR_CHARGE
import numpy as np

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
    
def mktmpdir():
    return subprocess.check_output(["mktemp", "-d", "-p", "."], text=True).rstrip("\n")
   
def mkdir(dir_name):
    subprocess.run(["mkdir", "-p", dir_name])
 
def rmdir(dirname):
    subprocess.run(["rm", "-Rf", dirname])

#   XYZ processing.
def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, compound.atomtypes, xyz_file_name)

def write_xyz_file(coordinates, elements, xyz_file_name):
    xyz_file=open(xyz_file_name, 'w')
    xyz_file.write(str(len(coordinates))+'\n\n')
    for atom_coords, element in zip(coordinates, elements):
        xyz_file.write(element+" "+' '.join([str(atom_coord) for atom_coord in atom_coords])+'\n')
    xyz_file.close()

def read_xyz(xyz_obj):
    if hasattr(xyz_obj, 'readlines'):
        return read_xyz_input(xyz_obj)
    else:
        return read_xyz_file(xyz_obj)

def read_xyz_file(xyz_file_name):
    xyz_input=open(xyz_file_name, "r")
    output=read_xyz_input(xyz_input)
    xyz_input.close()
    return output

def read_xyz_input(xyz_input):
    xyz_lines=xyz_input.readlines()
    natoms=int(xyz_lines[0])
    coordinates=np.zeros((natoms, 3))
    nuclear_charges=np.zeros((natoms, ), dtype=int)
    atomtypes=[]
    for atom_id, line in enumerate(xyz_lines[2:natoms+2]):
        spl_line=line.split()
        atomtype=canonical_atomtype(spl_line[0])
        atomtypes.append(atomtype)
        nuclear_charges[atom_id]=NUCLEAR_CHARGE[atomtype]
        for coord_id, coord in enumerate(spl_line[1:4]):
            coordinates[atom_id, coord_id]=float(coord)
    return atomtypes, nuclear_charges, coordinates

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

