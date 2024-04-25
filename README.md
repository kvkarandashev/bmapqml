# bmapqml

Collection of Quantum Machine Learning methods developed in University of Vienna during its participation in the [BIG-MAP consortium](https://www.big-map.eu). The project started off as a fork of the [QML code](https://github.com/qmlcode/qml). Most of the useful parts of the code can be found in [`mosaics`](https://github.com/chemspacelab/mosaics) and [`qml2`](https://github.com/chemspacelab/qml2) repositories.

With permission of Dominik Lemm, the repo also includes his Graph2Structure (`g2s`) code.

# Dependencies

pip-managed packages needed for the entire repository can be found in `bmapqml/requirements.txt`. For a given subfolder there could be additional requirements that are found in `requirements.txt` in the given subfolder.

Required packages not managed by pip and thus not mentioned in `requirements.txt` file are:
- for chemxpl : `xyz2mol`, `rdkit`, and `g2s` (the latter present in the repo)

# Installation.

After installing the dependencies modify `PYTHONPATH` environment variable to include the root directory of this repository or use `setup.py`.

# WARNING

Fortran parts of the code are compiled during first import of their Python interfaces, not installation. The procedure is experimental.
