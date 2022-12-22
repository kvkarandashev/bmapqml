# Show a bit what is inside the Docker.
import subprocess

subprocess.run(["pip", "list"])
subprocess.run(["python", "--version"])
subprocess.run(["conda", "--version"])

subprocess.run(["ls", "/"])

subprocess.run(["ls", "/extra_modules"])

# Some problematic imports.
from ase import Atoms
from dscribe.descriptors import SOAP

# from qml.representations import *
# from qml.kernels import get_local_kernel
