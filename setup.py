from setuptools import setup
import os
from subprocess import run

copied_endings = ["f90", "so"]


def attempt_import(full_filename):
    current_module_path = os.path.dirname(__file__)
    full_submodule_name = ".".join(full_filename.split(".")[0].split("/"))
    script_string = (
        """
export PYTHONPATH=\""""
        + current_module_path
        + """:$PYTHONPATH\"
python -c \"import """
        + full_submodule_name
        + """\"
"""
    )
    temp_script = "precomp_attempt.sh"
    with open(temp_script, "w") as f:
        print(script_string, file=f)
    run(["chmod", "+x", temp_script])
    run(["bash", temp_script])
    run(["rm", "-f", temp_script])


def attempt_precompile(full_filename):
    """
    If a filename contains reference to "precompiled" attempt importing it.
    """
    with open(full_filename, "r") as f:
        for line in f:
            line_spl = line.split()
            if not line_spl:
                continue
            first_el_spl = line_spl[0].split("(")
            if not first_el_spl:
                continue
            if first_el_spl[0] == "precompiled":
                attempt_import(full_filename)
                return


def packages_with_data_kwargs():
    unchecked_dirs = ["bmapqml"]
    package_data = {}
    while len(unchecked_dirs) != 0:
        checking_directory = unchecked_dirs[0]
        contains_data = []
        include = False
        for subdir in os.listdir(checking_directory):
            full_subdir = checking_directory + "/" + subdir
            if subdir == "__init__.py":
                include = True
                continue
            if os.path.isdir(full_subdir):
                if subdir == "__pycache__":
                    continue
                else:
                    unchecked_dirs.append(full_subdir)
            else:
                ending = subdir.split(".")[-1]
                if ending in copied_endings:
                    contains_data.append(subdir)
                    continue
                if ending == "py":
                    attempt_precompile(full_subdir)
        if include:
            package_data[checking_directory.replace("/", ".")] = contains_data
        del unchecked_dirs[0]
    return {"package_data": package_data, "packages": list(package_data.keys())}


if __name__ == "__main__":
    setup(**packages_with_data_kwargs())
