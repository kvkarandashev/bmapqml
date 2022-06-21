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

from joblib import Parallel, delayed
import os, subprocess
from .utils import mktmpdir, rmdir, dump2pkl, loadpkl

num_procs_name="BMAPQML_NUM_PROCS"

num_threads_var_names=["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]

def default_num_procs(num_procs=None):
    if num_procs is None:
        try:
            return int(os.environ[num_procs_name])
        except LookupError:
            return 1
    else:
        return num_procs

def create_func_exec(dump_filename, tmpdir, fixed_num_threads=1, num_procs=default_num_procs(None)):

    contents="#!/bin/bash\n"
    for num_threads_var_name in num_threads_var_names:
        contents+="export "+num_threads_var_name+"="+str(fixed_num_threads)+"\n"
    contents+="export "+num_procs_name+"="+str(num_procs)+"\n"
    contents+='''
pyscript='''+tmpdir+'''/tmppyscript.py
dump_filename='''+dump_filename+'''

cat > $pyscript << EOF
from bmapqml.python_parallelization import embarrassingly_parallel_no_thread_fix
from bmapqml.utils import loadpkl, dump2pkl

executed=loadpkl("$dump_filename")

dump_filename="$dump_filename"

output=embarrassingly_parallel_no_thread_fix(executed["func"], executed["array"], executed["args"])
dump2pkl(output, dump_filename)

EOF

python $pyscript'''
    exec_scr="./"+tmpdir+"/tmpscript.sh"
    output=open(exec_scr, 'w')
    output.write(contents)
    output.close()
    subprocess.run(["chmod", "+x", exec_scr])
    return exec_scr

def embarrassingly_parallel(func, array, other_args, other_kwargs={}, num_procs=None, fixed_num_threads=None):
    if type(other_args) is not tuple:
        other_args=(other_args,)
    if (fixed_num_threads is not None):
        if fixed_num_threads is None:
            true_num_threads=1
        else:
            true_num_threads=fixed_num_threads
        if num_procs is None:
            true_num_procs=default_num_procs(num_procs=num_procs)
        else:
            true_num_procs=num_procs
        tmpdir=mktmpdir()
        dump_filename=tmpdir+"/dump.pkl"
        dump2pkl({"func" : func, "array" : array, "args" : other_args}, dump_filename)
        exec_scr=create_func_exec(dump_filename, tmpdir, fixed_num_threads=true_num_threads, num_procs=true_num_procs)
        subprocess.run([exec_scr, dump_filename])
        output=loadpkl(dump_filename)
        rmdir(tmpdir)
        return output
    else:
        if num_procs == 1:
            return [func(element, *other_args, **other_kwargs) for element in array]
        else:
            return embarrassingly_parallel_no_thread_fix(func, array, other_args, kwargs=other_kwargs, num_procs=num_procs)

def embarrassingly_parallel_no_thread_fix(func, array, args, num_procs=None, kwargs={}):
    return Parallel(n_jobs=default_num_procs(num_procs), backend="multiprocessing")(delayed(func)(el, *args, **kwargs) for el in array)
