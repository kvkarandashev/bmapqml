from bmapqml.chemxpl.rdkit_utils import chemgraph_to_canonical_rdkit
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
import random

random.seed(1)

s = "7@1@2:6@3@4:6#1@3:6#1:6#3"
for _ in range(10):
    cg = str2ChemGraph(s, shuffle=True)
    print(chemgraph_to_canonical_rdkit(cg, SMILES_only=True))
