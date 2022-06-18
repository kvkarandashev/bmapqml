
from bmapqml.utils import analyze_random_walk
import numpy as np
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import pdb

parser = argparse.ArgumentParser(description='INPUT PATH')
parser.add_argument("-PATH")
parser.add_argument("-plotname", default="plot")
args = parser.parse_args()

path = args.PATH
plotname=args.plotname
WEIGHTS = np.array([ (1/1.9), (1/6.8)])

min_func = multi_obj(
    [QM9_properties("/store/common/jan/qm9/KRR_12000_atomization",verbose=False),QM9_properties("/store/common/jan/qm9/KRR_12000_gap",verbose=False)
], WEIGHTS, verbose=True)


ana = analyze_random_walk("{}".format(path)+"QM9_histogram.pkl","{}".format(path)+"QM9_best_candidates.pkl", model=min_func)

values = ana.evaluate_histogram()
front, inds, mols = ana.compute_pareto_front()
best_P1, best_P2, best_loss = np.argsort(front[:,0]),np.argsort(front[:,1]),np.argsort(values[:,2][:len(front)])

print("Best Candidates:")
[print(m) for m in mols[best_loss]]
print("Most Stable:")
[print(m, "{:.2f}".format(P1)+" eV") for m,P1 in zip(mols[best_P1], front[:,0][best_P1])] 
print("Smallest Gap:")
[print(m,"{:.2f}".format(P2)+" eV") for m,P2 in zip(mols[best_P2], front[:,1][best_P2])] 

sns.set_context("poster")
sns.set_style('whitegrid')
fs = 24
markersize = 12
alpha=0.65

plt.rc('font', size=fs)
plt.rc('axes', titlesize=fs)
plt.rc('axes', labelsize=fs)           
plt.rc('xtick', labelsize=fs)          
plt.rc('ytick', labelsize=fs)          
plt.rc('legend', fontsize=fs)   
plt.rc('figure', titlesize=fs) 
fig,ax1= plt.subplots(figsize=(8,8))

p1  = values[:,0]
p2  = values[:,1]
summe = values[:,2]
xi = np.linspace(min(p1), max(p1), 100)
yi = np.linspace(min(p2), max(p2), 100)

triang = tri.Triangulation(p1, p2)
interpolator = tri.LinearTriInterpolator(triang,summe)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

ax1.contour(xi, yi, zi, levels=18, linewidths=0.5, colors='k')
sc = ax1.scatter(p1, p2,s =4, c=summe)
plt.xlabel("$E_{\\rm {at}}/ N^{\\rm tot}$"  + " [eV]", fontsize=21)
plt.ylabel("$E_{\\rm {gap}}$" + " [eV]", fontsize=21)
clb = plt.colorbar(sc)
clb.set_label("Loss") 

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['bottom'].set_position(('axes', -0.05))
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['left'].set_position(('axes', -0.05))

plt.plot(front[:,0],front[:,1],'o', color='black')
plt.plot(front[:,0],front[:,1],'k-',linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("{}.pdf".format(plotname))




"""
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from mpld3 import plugins
mpld3.enable_notebook()

def moltosvg(mol,molSize=(450,15),kekulize=True):
    mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')


    #svgs = [moltosvg(m) for m in mols]

mols = np.array(ana.histogram)
svgs = [moltosvg(m) for m in mols]
fig2,ax2= plt.subplots(figsize=(8,8))
points = ax2.scatter(front[:,0],front[:,1]) 
tooltip = plugins.PointHTMLTooltip(points, svgs)
plugins.connect(fig2, tooltip)
"""

