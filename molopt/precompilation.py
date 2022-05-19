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

# TO-DO Not %100 sure that the algorithm deciding the compilation order is fool proof. Might need extension.

import subprocess, importlib, os

from .utils import execute_string

debug_mode_envvar="MOLOPT_FORTAN_DEBUG"

library_preprocessor_flags={"!$OMP" : "OpenMP"}

library_args={"OpenMP" : ["-lgomp"]}

library_compiler_flags={"OpenMP" : ["-fopenmp"]}



def append_unrepeated(l1, l2):
    for el in l2:
        add_unrepeated(l1, el)

def add_unrepeated(l, obj):
    if obj not in l:
        l.append(obj)

class Source:
    def __init__(self, module_name, source_depending=[], parent_entry=None):
        self.module_name=module_name
        self.source_name=None
        self.source_dir=None
        self.library_dependencies=None

        self.module_dependencies=None
        self.source_dependencies=None
        self.source_depending=source_depending

        self.parent_entry=parent_entry
    def find_source_in_dirs(self, searched_dirs, extensions, lowest_dir=None):
        if lowest_dir is not None:
            started=False

        for searched_dir in searched_dirs:
            if lowest_dir is not None:
                if lowest_dir == searched_dir:
                    started=True
                if not started:
                    continue
            for extension in extensions:
                tried=searched_dir+"/"+self.module_name+"."+extension
                if os.path.isfile(tried):
                    self.source_name=tried
                    self.source_dir=searched_dir
                    return
        raise Exception("Source not found.")

    def find_dependencies(self, searched_dirs, syntax="Fortran", extensions=["f90", "f"]):
        source_input=open(self.source_name, "r")
        source_lines=source_input.readlines()
        source_input.close()

        self.module_dependencies=[]
        self.source_dependencies=[]
        self.library_dependencies=[]

        for source_line in source_lines:
            source_line_spl=source_line.split()
            self.add_module_dependency(source_line_spl)
            self.add_library_dependency(source_line_spl)

        new_source_depending_list=[self.source_name]+self.source_depending
        for md in self.module_dependencies:
            new_source=Source(md, source_depending=new_source_depending_list)
            new_source.find_source_in_dirs(searched_dirs, extensions, lowest_dir=self.source_dir)
            self.source_dependencies.append(new_source)
    def add_module_dependency(self, source_line_spl):
        if len(source_line_spl)!=0:
            if source_line_spl[0]=="use":
                new_mod=source_line_spl[1].split(",")[0]
                add_unrepeated(self.module_dependencies, new_mod)
    def add_library_dependency(self, source_line_spl):
        if len(source_line_spl)!=0:
            for lpf, lib in library_preprocessor_flags.items():
                if source_line_spl[0]==lpf:
                    add_unrepeated(self.library_dependencies, lib)

    def add_to_source_depending(self, new_sources):
        for ns in new_sources:
            add_unrepeated(self.source_depending, ns)

    def __gt__(self, other):
        return (self.source_name in other.source_depending)
    def __eq__(self, other):
        if isinstance(other, str):
            return (self.source_name==other)
        else:
            return (self.source_name==other.source_name)
    def __str__(self):
        if self.source_name is None:
            output="source_undefined,module:"+self.module_name
        else:
            output=self.source_name
        if self.source_dependencies is not None:
            output+=";dependencies:"
            for s in self.source_dependencies:
                output+=str(s)+","
        if self.source_depending is not None:
            output+=";depending:"
            for s in self.source_depending:
                output+=str(s)+","
        if self.parent_entry is not None:
            output+=";parent entry:"+self.parent_entry
        return output
    def __repr__(self):
        return str(self)

def module_source_dirs(module_path, file_in_module_root_dir):
    mp_split=module_path.split(".")
    child_dirs=module_path.split(".")[:-1]
    output=[os.path.dirname(__file__)]
    for chdir in child_dirs:
        output.append(output[-1]+"/"+chdir)
    return mp_split[-1], output[::-1]

def add_f2py_args(library_dependencies):
    add_flags_list=["-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
    if ((debug_mode_envvar in os.environ) and (os.environ[debug_mode_envvar] == "1")):
        # Run the code in debug mode.
        add_flags_list+=["-g", "-fcheck=all", "-Wall"]
    else:
        # Run optimized version of the code
        add_flags_list+=["-O3"]
    add_args=[]
    for ld in library_dependencies:
        add_args+=library_args[ld]
        add_flags_list+=library_compiler_flags[ld]
    add_flags_arg="--f90flags='"
    for add_flag in add_flags_list:
        add_flags_arg+=add_flag+" "
    add_flags_arg=add_flags_arg[:-1]+"'"
    return (add_flags_arg, *add_args)


def precompiled(*module_paths, extensions=["f90", "f"], parent_module_name="molopt", file_in_module_root_dir=__file__):
    for module_path in module_paths:
        try:
            importlib.import_module(parent_module_name+"."+module_path)
        except ModuleNotFoundError:
            module_name, searched_dirs=module_source_dirs(module_path, file_in_module_root_dir)
            library_dependencies=[]
            compiled_sources=[Source(module_name)]
            cur_source_id=0
            compiled_sources[cur_source_id].find_source_in_dirs(searched_dirs, extensions)
            while cur_source_id != len(compiled_sources):
                print(compiled_sources, cur_source_id)
                cur_source=compiled_sources[cur_source_id]
                if cur_source in compiled_sources[:cur_source_id]:
                    duplicated_source_id=compiled_sources[:cur_source_id].index(cur_source)
                    append_unrepeated(compicated_sources[duplicated_source_id].source_depending, cur_source.source_depending)
                    for other_source_id in range(cur_source_id):
                        if compiled_sources[other_source_id].parent_entry is not None:
                            if compiled_sources[other_source_id].parent_entry == cur_source.source_name:
                                append_unrepeated(compicated_sources[duplicated_source_id].source_depending, cur_source.source_depending)
                    del(compiled_sources[cur_source_id])
                else:
                    cur_source.find_dependencies(searched_dirs, extensions=extensions)
                    append_unrepeated(compiled_sources, cur_source.source_dependencies)
                    append_unrepeated(library_dependencies, cur_source.library_dependencies)
                    cur_source_id+=1

            compiled_sources.sort()
            calldir=os.getcwd()
            os.chdir(searched_dirs[0])
            final_command="f2py -c"
            for added_arg in add_f2py_args(library_dependencies):
                final_command+=" "+added_arg
            final_command+=" -m "+module_name
            for compiled_source in compiled_sources:
                final_command+=" "+compiled_source.source_name
            execute_string(final_command)
            os.chdir(calldir)
