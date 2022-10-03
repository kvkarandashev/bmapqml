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
import pickle, subprocess, os
import bz2
from joblib import Parallel, delayed

from .data import NUCLEAR_CHARGE
import numpy as np

# Some auxiliary functions.
def np_resize(np_arr, new_size):
    """
    Expand or cut a NumPy array.
    """
    new_arr = np_arr
    for dim_id, new_dim in enumerate(new_size):
        cur_dim = new_arr.shape[dim_id]
        if new_dim is None:
            continue
        if cur_dim < new_dim:
            add_arr_dims = list(new_arr.shape)
            add_arr_dims[dim_id] = new_dim - cur_dim
            add_arr = np.zeros(tuple(add_arr_dims))
            new_arr = np.append(new_arr, add_arr, dim_id)
        if cur_dim > new_dim:
            new_arr = np.delete(new_arr, slice(new_dim, cur_dim), dim_id)
    return new_arr


def canonical_atomtype(atomtype):
    return atomtype[0].upper() + atomtype[1:].lower()


def any_element_in_list(list_in, *els):
    for el in els:
        if el in list_in:
            return True
    return False


def repeated_dict(labels, repeated_el):
    output = {}
    for l in labels:
        output[l] = repeated_el
    return output


def all_None_dict(labels):
    return repeated_dict(labels, None)


ELEMENTS = None


def str_atom_corr(ncharge):
    global ELEMENTS
    if ELEMENTS is None:
        ELEMENTS = {}
        for cur_el, cur_ncharge in NUCLEAR_CHARGE.items():
            ELEMENTS[cur_ncharge] = cur_el
    return ELEMENTS[ncharge]


# def str_atom_corr(ncharge):
#    return canonical_atomtype(str_atom(ncharge))


def dump2pkl(obj, filename):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    """
    output_file = open(filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadpkl(filename):
    """
    Load an object from a pickle file.
    """
    input_file = open(filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def dump2tar(obj, filename):
    """
    Dump an object to a tar file.
    obj : object to be saved
    filename : name of the output file
    """
    output_file = bz2.BZ2File(filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadtar(filename):
    """
    Load an object from a tar file.
    """
    input_file = bz2.BZ2File(filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def mktmp(directory=False):
    extra_args = ()
    if directory:
        extra_args = ("-d", *extra_args)
    return subprocess.check_output(
        ["mktemp", *extra_args, "-p", "."], text=True
    ).rstrip("\n")


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
        return byte_or_str.decode("utf-8")


#   For processing xyz files.
def checked_input_readlines(file_input):
    try:
        lines = [check_byte(l) for l in file_input.readlines()]
    except AttributeError:
        with open(file_input, "r") as input_file:
            lines = input_file.readlines()
    return lines


def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, xyz_file_name, elements=compound.atomtypes)


def xyz_string(coordinates, elements=None, nuclear_charges=None):
    """
    Create an xyz-formatted string from coordinates and elements or nuclear charges.
    coordinates : coordinate array
    elements : string array of element symbols
    nuclear_charges : integer array; used to generate element list if elements is set to None
    """
    if elements is None:
        elements = [str_atom_corr(charge) for charge in nuclear_charges]
    output = str(len(coordinates)) + "\n"
    for atom_coords, element in zip(coordinates, elements):
        output += (
            "\n"
            + element
            + " "
            + " ".join([str(atom_coord) for atom_coord in atom_coords])
        )
    return output


def write_xyz_file(coordinates, xyz_file_name, elements=None, nuclear_charges=None):
    xyz_file = open(xyz_file_name, "w")
    xyz_file.write(
        xyz_string(coordinates, elements=elements, nuclear_charges=nuclear_charges)
    )
    xyz_file.close()


def read_xyz_file(xyz_input, additional_attributes=["charge"]):

    lines = checked_input_readlines(xyz_input)

    return read_xyz_lines(lines, additional_attributes=additional_attributes)


def read_xyz_lines(xyz_lines, additional_attributes=["charge"]):
    add_attr_dict = {}
    for add_attr in additional_attributes:
        add_attr_dict = {add_attr: None, **add_attr_dict}

    num_atoms = int(xyz_lines[0])
    xyz_coordinates = np.zeros((num_atoms, 3))
    nuclear_charges = np.zeros((num_atoms,), dtype=int)
    atomic_symbols = []

    lsplit = xyz_lines[1].split()
    for l in lsplit:
        for add_attr in additional_attributes:
            add_attr_eq = add_attr + "="
            if add_attr_eq == l[: len(add_attr_eq)]:
                add_attr_dict[add_attr] = int(l.split("=")[1])

    for atom_id, atom_line in enumerate(xyz_lines[2 : num_atoms + 2]):
        lsplit = atom_line.split()
        atomic_symbol = lsplit[0]
        atomic_symbols.append(atomic_symbol)
        nuclear_charges[atom_id] = NUCLEAR_CHARGE[canonical_atomtype(atomic_symbol)]
        for i in range(3):
            xyz_coordinates[atom_id, i] = float(lsplit[i + 1])

    return nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict


def xyz_file_stochiometry(xyz_input, by_atom_symbols=True):
    """
    Stochiometry of the xyz input
    xyz_input : either name of an xyz file or the corresponding _io.TextIOWrapper instance
    by_atom_symbols : if "True" use atomic symbols as keys, otherwise use nuclear charges.
    """
    nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict = read_xyz_file(
        xyz_input, additional_attributes=[]
    )
    if by_atom_symbols:
        identifiers = atomic_symbols
    else:
        identifiers = nuclear_charges
    output = {}
    for i in identifiers:
        if i in output:
            output[i] += 1
        else:
            output[i] = 1
    return output


def write2file(string, file_name):
    file_output = open(file_name, "w")
    print(string, file=file_output)
    file_output.close()


class OptionUnavailableError(Exception):
    pass


def merged_representation_array(total_compound_array):
    return [
        np.array(
            [
                total_compound_array[comp_id].representation
                for comp_id in range(*index_tuple)
            ]
        )
    ]


def where2slice(indices_to_ignore):
    return np.where(np.logical_not(indices_to_ignore))[0]


def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(np.logical_not(cur_ignore_indices))] = 0.0


#   A dumb way to run commands without regards for spaces inside them when subprocess.run offers no viable workarounds..
def execute_string(string):
    script_name = mktmp()
    subprocess.run(["chmod", "+x", script_name])
    script_output = open(script_name, "w")
    script_output.write("#!/bin/bash\n" + string)
    script_output.close()
    subprocess.run(["chmod", "+x", script_name])
    subprocess.run(["./" + script_name])
    rmdir(script_name)


# Routines for running operations over array in child environments.
# One of the reason for this appearing is fixing thread numbers in child processes.
class ChildEnvFailed(Exception):
    pass


num_procs_name = "BMAPQML_NUM_PROCS"

num_threads_var_names = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
]


def checked_environ_val(
    environ_name: str, expected_answer=None, default_answer=None, var_class=int
):
    """
    Returns os.environ while checking for exceptions.
    """
    if expected_answer is None:
        try:
            args = (os.environ[environ_name],)
        except LookupError:
            if default_answer is None:
                args = tuple()
            else:
                args = (default_answer,)
        return var_class(*args)
    else:
        return expected_answer


def default_num_procs(num_procs=None):
    return checked_environ_val(
        num_procs_name, expected_answer=num_procs, default_answer=1
    )


def embarrassingly_parallel_no_thread_fix(
    func, array, other_args, other_kwargs={}, num_procs=None
):
    if num_procs == 1:
        return [func(element, *other_args, **other_kwargs) for element in array]
    else:
        return Parallel(n_jobs=default_num_procs(num_procs), backend="multiprocessing")(
            delayed(func)(el, *other_args, **other_kwargs) for el in array
        )


def create_func_exec(dump_filename, tmpdir, fixed_num_threads=None, num_procs=None):

    contents = "#!/bin/bash\n"
    if fixed_num_threads is not None:
        for num_threads_var_name in num_threads_var_names:
            contents += (
                "export " + num_threads_var_name + "=" + str(fixed_num_threads) + "\n"
            )
    if num_procs is not None:
        contents += "export " + num_procs_name + "=" + str(num_procs) + "\n"
    contents += (
        """
pyscript="""
        + tmpdir
        + """/tmppyscript.py
dump_filename="""
        + dump_filename
        + """

cat > $pyscript << EOF
from bmapqml.utils import loadpkl, dump2pkl, rmdir, embarrassingly_parallel_no_thread_fix

executed=loadpkl("$dump_filename")
rmdir("$dump_filename")

dump_filename="$dump_filename"

output=embarrassingly_parallel_no_thread_fix(executed["func"], executed["array"], executed["args"])
dump2pkl(output, dump_filename)

EOF

python $pyscript"""
    )
    exec_scr = "./" + tmpdir + "/tmpscript.sh"
    output = open(exec_scr, "w")
    output.write(contents)
    output.close()
    subprocess.run(["chmod", "+x", exec_scr])
    return exec_scr


def create_run_func_exec(
    func, array, other_args, other_kwargs, num_procs=None, fixed_num_threads=None
):

    tmpdir = mktmpdir()
    dump_filename = tmpdir + "/dump.pkl"
    dump2pkl({"func": func, "array": array, "args": other_args}, dump_filename)
    exec_scr = create_func_exec(
        dump_filename, tmpdir, num_procs=num_procs, fixed_num_threads=fixed_num_threads
    )
    subprocess.run([exec_scr, dump_filename])
    if not os.path.isfile(dump_filename):
        raise ChildEnvFailed
    output = loadpkl(dump_filename)
    rmdir(tmpdir)
    return output


def embarrassingly_parallel(
    func, array, other_args, other_kwargs={}, num_procs=None, fixed_num_threads=None
):
    if type(other_args) is not tuple:
        other_args = (other_args,)
    if fixed_num_threads is not None:
        if num_procs is None:
            true_num_procs = default_num_procs(num_procs=num_procs)
        else:
            true_num_procs = num_procs
        return create_run_func_exec(
            func,
            array,
            other_args,
            other_kwargs,
            num_procs=true_num_procs,
            fixed_num_threads=fixed_num_threads,
        )
    else:
        return embarrassingly_parallel_no_thread_fix(
            func, array, other_args, other_kwargs=other_kwargs, num_procs=num_procs
        )


#   Run a function that might send a termination signal rather than raise an exception.
def safe_array_func_eval(
    func,
    array,
    other_args,
    other_kwargs={},
    num_procs=1,
    fixed_num_threads=None,
    failure_placeholder=None,
):
    try:
        return create_run_func_exec(
            func,
            array,
            other_args,
            other_kwargs,
            num_procs=num_procs,
            fixed_num_threads=fixed_num_threads,
        )
    except ChildEnvFailed:
        if len(array) == 1:
            return [failure_placeholder]
        else:
            divisor = len(array) // 2
            output = []
            for l in [array[:divisor], array[divisor:]]:
                output += safe_array_func_eval(
                    func,
                    l,
                    other_args,
                    other_kwargs={},
                    num_procs=num_procs,
                    fixed_num_threads=fixed_num_threads,
                    failure_placeholder=failure_placeholder,
                )
            return output
