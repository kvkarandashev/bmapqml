from setuptools import setup
import os

copied_endings = ["f90"]


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
                if subdir.split(".")[-1] in copied_endings:
                    contains_data.append(subdir)
        if include:
            package_data[checking_directory.replace("/", ".")] = contains_data

        del unchecked_dirs[0]
    return {"package_data": package_data, "packages": list(package_data.keys())}


if __name__ == "__main__":
    setup(**packages_with_data_kwargs())
