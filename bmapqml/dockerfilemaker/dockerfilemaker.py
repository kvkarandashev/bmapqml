import glob, os, subprocess
from ..utils import mktmpdir

# Filename suffix for files containing format specifications.
dockerspec_filename_suffix = ".docker_spec"

special_cond_separator = ";"

"""
Flags used to specify what should be included into the docker file. Flags after 'PARENT' are written in order in which commands are added to Dockerfile:
PARENT      include everything from another docker_spec file. "Circular parenting" does not crash the code.
FROM        which Docker to start from ("child" supercedes "parent")
APT         packages that should be installed with apt.
CONDAV      version of CONDA to be installed (currently does not work, perhaps I don't know enough about conda).
            Entry in the "child" takes priority over the entry in the parent.
PYTHONV     version of python; "child" takes priority.
PIP         packages installed with pip. Additional necessary flags can be put with special_cond_separator.
PIPLAST     packages that should be additionally installed with PIP (e.g. qml's setup.py requires numpy to run, producing bug if put into the 'PIP' section).
CONDA       packages that should be installed with conda; provided in '${package name}${special_cond_separator}${conda channel}' format.
PYTHONPATH  python modules installed by copy to PYTHONPATH.
"""

parent_flag = "PARENT"
pythonpath_flag = "PYTHONPATH"
apt_flag = "APT"
from_flag = "FROM"

dependency_specifiers = [
    from_flag,
    apt_flag,
    "CONDAV",
    "PYTHONV",
    "PIP",
    "PIPLAST",
    "CONDA",
    pythonpath_flag,
]

contains_miniconda = ["continuumio/miniconda3:4.10.3-alpine"]

# Where most basic commands for Docker are found.
base_dockerfile_cmd_file = "base_dockerfile_commands.txt"

# Command for normal shell operation.
login_shell_command = 'SHELL ["/bin/bash", "--login", "-c"]'
# Command for root (?) shell operation. Perhaps not needed?
root_shell_command = ""  #'SHELL ["/bin/bash", "-c"]'


def available_dockers_dir():
    """
    The directory where Docker specification files available by default are stored.
    """
    return os.path.dirname(__file__)


def available_dockers():
    """
    Docker specification files available by default.
    """
    return [
        spec_file[:-4]
        for spec_file in glob.glob(
            available_dockers_dir() + "/*" + dockerspec_filename_suffix
        )
    ]


def dockerspec_filename(docker_name, dockerspec_dir=available_dockers_dir()):
    """
    Docker specification file corresponding to a Docker file to be created.
    """
    return dockerspec_dir + "/" + docker_name + dockerspec_filename_suffix


# Command for updating conda.
conda_update_command = "RUN conda update -n base conda"
conda_installation_script_name = "Miniconda3-latest-Linux-x86_64.sh"
internal_installation_files = "/installation_files"
internal_conda_dir = "/opt/conda"


def conda_installation_lines(temporary_folder):
    """
    If we need to install conda inside a Docker container we add these lines to the Dockerfile script.
    """
    # Solution based on https://fabiorosado.dev/blog/install-conda-in-docker/
    subprocess.run(
        [
            "wget",
            "--quiet",
            "https://repo.anaconda.com/miniconda/" + conda_installation_script_name,
            "-O",
            temporary_folder + "/" + conda_installation_script_name,
        ]
    )
    internal_conda_install = (
        internal_installation_files + "/" + conda_installation_script_name
    )
    output = [
        "COPY "
        + temporary_folder
        + "/"
        + conda_installation_script_name
        + " "
        + internal_conda_install
    ]
    output += [
        "RUN chmod +x " + internal_conda_install,
        "RUN " + internal_conda_install + " -b -p " + internal_conda_dir,
        "ENV PATH=" + internal_conda_dir + "/bin:$PATH",
    ]
    return output


def get_from_dep_lines(dep_list):
    return ["FROM " + dep_list[0]]


def get_apt_dep_lines(dep_list):
    l = "RUN apt-get install -y"
    for dep in dep_list:
        l += " " + dep
    return [root_shell_command, "RUN apt-get update", l]


def get_local_dependencies(docker_name, dockerspec_dir=available_dockers_dir()):
    spec_filename = dockerspec_filename(docker_name, dockerspec_dir=dockerspec_dir)
    if not os.path.isfile(spec_filename):
        raise Exception("Unknown Docker spec filename.")
    processed_lines = open(spec_filename, "r").readlines()
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
        dep_spl = dep.split(special_cond_separator)
        package_name = dep_spl[0]
        if len(dep_spl) > 1:
            channel_name = dep_spl[1]
            channel_args = "-c " + channel_name + " "
        else:
            channel_args = ""
        output.append("RUN conda install " + channel_args + package_name)
    return output


def pip_install_line(comp):
    l = "RUN pip install"
    for c in comp:
        l += " " + c
    return l


def get_pip_dep_lines(dep_list):
    no_special_flags = []
    wspecial_flags = []
    for dep in dep_list:
        if special_cond_separator in dep:
            wspecial_flags.append(dep)
        else:
            no_special_flags.append(dep)
    lines = []
    if no_special_flags:
        lines.append(pip_install_line(no_special_flags))
    if wspecial_flags:
        for dep in wspecial_flags:
            dep_spl = dep.split(special_cond_separator)
            lines.append(pip_install_line(dep_spl))
    return lines


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


dependency_line_dict = {
    from_flag: get_from_dep_lines,
    apt_flag: get_apt_dep_lines,
    "PIP": get_pip_dep_lines,
    "PIPLAST": get_pip_dep_lines,
    "CONDA": get_conda_dep_lines,
    pythonpath_flag: get_pythonpath_dep_lines,
    "CONDAV": get_conda_version_specification,
    "PYTHONV": get_python_version_specification,
}


def get_all_dependencies(docker_name, dockerspec_dir=available_dockers_dir()):
    cur_imported_id = 0
    dep_dict = {parent_flag: [docker_name]}
    while cur_imported_id != len(dep_dict[parent_flag]):
        to_add = get_local_dependencies(
            dep_dict[parent_flag][cur_imported_id], dockerspec_dir=dockerspec_dir
        )
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
    dockerspec_dir=available_dockers_dir(),
    conda_updated=True,
):
    base_dockerfile_cmd_full_path = dockerspec_dir + "/" + base_dockerfile_cmd_file
    # Commands basic for leruli-compliant Docker.
    assert os.path.isfile(base_dockerfile_cmd_full_path)
    # Temporary directory where necessary files will be dumped.
    temp_dir = mktmpdir()
    # Docker-specific dependencies.
    all_dependencies = get_all_dependencies(docker_name, dockerspec_dir=dockerspec_dir)
    if from_flag not in all_dependencies:
        raise Exception()
    output = get_from_dep_lines(all_dependencies[from_flag])
    output += open(base_dockerfile_cmd_full_path, "r").readlines()

    # Commands run once we set up apt-installable components.
    post_apt_commands = [login_shell_command]
    if all_dependencies[from_flag][0] not in contains_miniconda:
        post_apt_commands += conda_installation_lines(temp_dir)

    if conda_updated:
        post_apt_commands.append(conda_update_command)

    if apt_flag not in all_dependencies:
        output += post_apt_commands

    for dep_spec in dependency_specifiers[1:-1]:
        if dep_spec not in all_dependencies:
            continue
        output += dependency_line_dict[dep_spec](all_dependencies[dep_spec])
        if dep_spec == apt_flag:
            output += post_apt_commands

    if pythonpath_flag in all_dependencies:
        copy_reqs = all_dependencies[pythonpath_flag]
        output += dependency_line_dict[pythonpath_flag](
            all_dependencies[pythonpath_flag], temp_dir
        )
    else:
        copy_reqs = []
        temp_dir = None
    return output, copy_reqs, temp_dir


def prepare_dockerfile(docker_name, dockerspec_dir=available_dockers_dir()):
    dlines, copy_reqs, temp_dir = get_dockerfile_lines_deps(
        docker_name, dockerspec_dir=dockerspec_dir
    )
    output = open("Dockerfile", "w")
    for l in dlines:
        print(l, file=output)
    output.close()
    if temp_dir is not None:
        for copy_req in copy_reqs:
            subprocess.run(["cp", "-r", get_module_imported_dir(copy_req), temp_dir])
