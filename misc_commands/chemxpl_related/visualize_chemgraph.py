from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.rdkit_draw_utils import draw_chemgraph_to_file
import sys, os

chemgraph_str = sys.argv[1]

if len(sys.argv) > 2:
    filename_prefix = sys.argv[2]
else:
    filename_prefix = "chemgraph_visualization_"

cg = str2ChemGraph(chemgraph_str)

i = 0

while True:
    filename = filename_prefix + str(i) + ".png"
    if os.path.isfile(filename):
        i += 1
    else:
        draw_chemgraph_to_file(
            cg,
            filename,
            size=(300, 200),
        )
        break
