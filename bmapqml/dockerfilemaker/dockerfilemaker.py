import glob, os, subprocess
from ..utils import mktmpdir

dockerspec_filename_suffix = ".docker_spec"


# Flag used to specified one or more parents to the Docker. Circular parenting does not crash the code.
parent_flag = "PARENT"
pythonpath_flag = "PYTHONPATH"

dependency_specifiers = ["CONDAV", "PYTHONV", "PIP", "CONDA", pythonpath_flag]

base_dockerfile_cmd_file = "base_dockerfile_commands.txt"


def available_dockers_dir():
    return os.path.dirname(__file__)


def available_dockers():
    return [
        spec_file[:-4]
        for spec_file in glob.glob(
            available_dockers_dir() + "/*" + dockerspec_filename_suffix
        )
    ]


def dockerspec_filename(docker_name):
    return available_dockers_dir() + "/" + docker_name + dockerspec_filename_suffix


def get_local_dependencies(docker_name):
    spec_filename = dockerspec_filename(docker_name)
    if not os.path.isfile(spec_filename):
        raise Exception("Unknown Docker spec filename.")
    processed_lines = open(dockerspec_filename(docker_name), "r").readlines()
    output = {}
    for l in processed_lines:
        lspl = l.split()
        flag = lspl[0]
        if (flag in dependency_specifiers) or (flag == parent_flag):
            if flag not in output:
                output[flag] = []
            output[flag] += lspl[1:]
    return output


def get_conda_dep_lines(dep_list):
    output = []
    for dep in dep_list:
        dep_spl = dep.split(":")
        package_name = dep_spl[0]
        if len(dep_spl) > 1:
            channel_name = dep_spl[1]
            channel_args = "-c " + channel_name + " "
        else:
            channel_args = ""
        output.append("RUN conda install " + channel_args + package_name)
    return output


def get_pip_dep_lines(dep_list):
    l = "RUN pip install"
    for dep in dep_list:
        l += " " + dep
    return [l]


def get_module_imported_dir(module_name):
    initfile_command = "import " + module_name + "; print(" + module_name + ".__file__)"
    init_file = subprocess.run(
        ["python", "-c", initfile_command], capture_output=True
    ).stdout.decode("utf-8")
    return os.path.dirname(init_file)


def get_pythonpath_dep_lines(dep_list, temp_module_copy_dir):
    output = []
    for dep in dep_list:
        destination = "/extra_modules/" + dep
        output.append("COPY " + temp_module_copy_dir + "/" + dep + " " + destination)
        if not os.path.isfile(get_module_imported_dir(dep) + "/__init__.py"):
            output.append("ENV PYTHONPATH " + destination + ":$PYTHONPATH")
    return output


# TODO does not work for some reason.
def get_conda_version_specification(dep_list):
    return ["RUN conda install anaconda=" + dep_list[0]]


def get_python_version_specification(dep_list):
    return ["RUN conda install python=" + dep_list[0]]


# TODO add something for packages that can only be installed with conda.
dependency_line_dict = {
    "PIP": get_pip_dep_lines,
    "CONDA": get_conda_dep_lines,
    pythonpath_flag: get_pythonpath_dep_lines,
    "CONDAV": get_conda_version_specification,
    "PYTHONV": get_python_version_specification,
}


def get_all_dependencies(docker_name):
    cur_imported_id = 0
    dep_dict = {parent_flag: [docker_name]}
    while cur_imported_id != len(dep_dict[parent_flag]):
        to_add = get_local_dependencies(dep_dict[parent_flag][cur_imported_id])
        for dep_type, dep_list in to_add.items():
            if dep_type not in dep_dict:
                dep_dict[dep_type] = []
            for dep in dep_list:
                if dep not in dep_dict[dep_type]:
                    dep_dict[dep_type].append(dep)
        cur_imported_id += 1
    del dep_dict[parent_flag]
    return dep_dict


def get_dockerfile_lines_deps(
    docker_name,
):
    base_dockerfile_cmd_full_path = (
        available_dockers_dir() + "/" + base_dockerfile_cmd_file
    )
    assert os.path.isfile(base_dockerfile_cmd_full_path)
    output = open(base_dockerfile_cmd_full_path, "r").readlines()
    all_dependencies = get_all_dependencies(docker_name)
    for dep_spec in dependency_specifiers[:-1]:
        if dep_spec not in all_dependencies:
            continue
        output += dependency_line_dict[dep_spec](all_dependencies[dep_spec])
    if pythonpath_flag in all_dependencies:
        copy_reqs = all_dependencies[pythonpath_flag]
        temp_dir = mktmpdir()
        output += dependency_line_dict[pythonpath_flag](
            all_dependencies[pythonpath_flag], temp_dir
        )
    else:
        copy_reqs = []
        temp_dir = None
    return output, copy_reqs, temp_dir


def prepare_dockerfile(docker_name):
    dlines, copy_reqs, temp_dir = get_dockerfile_lines_deps(docker_name)
    output = open("Dockerfile", "w")
    for l in dlines:
        print(l, file=output)
    output.close()
    if temp_dir is not None:
        for copy_req in copy_reqs:
            subprocess.run(["cp", "-r", get_module_imported_dir(copy_req), temp_dir])
