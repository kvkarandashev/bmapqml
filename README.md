# molopt

Collection of methods developed as part of BIG-MAP project for Quantum Machine Learning. The project started off as a branch of the QML code.

(Currently the actual python package's name is "bmapqml," but this might change in the future. The repository name stays as "molopt" for now.)

With permission of Dominik Lemm, the repo also includes his Graph2Structure (g2s) code.

# Dependencies

pip-managed packages needed for the entire repository can be found in bmapqml/requirements.txt. For a given subfolder there could be additional requirements that are found in requirements.txt in the given subfolder.

Required packages not managed by pip and thus not mentioned in requirements.txt file are:
- for chemxpl : xyz2mol, rdkit, and g2s (the latter present in the repo)
