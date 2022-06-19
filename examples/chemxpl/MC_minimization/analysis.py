
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
parser.add_argument("-name", default="plot")
parser.add_argument("-properties", default="atomization_gap")
args = parser.parse_args()

path = args.PATH
plotname=args.name

if args.properties == "atomization_gap":
    WEIGHTS = np.array([ (1/1.9), (1/6.8)])
    min_func = multi_obj(
        [QM9_properties("/store/common/jan/qm9/KRR_12000_atomization",verbose=False),QM9_properties("/store/common/jan/qm9/KRR_12000_gap",verbose=False)
    ], WEIGHTS, verbose=True)

if args.properties == "gap":
    min_func = QM9_properties("/store/common/jan/qm9/KRR_12000_gap",verbose=False)

if args.properties == "atomization":
    min_func = QM9_properties("/store/common/jan/qm9/KRR_12000_atomization",verbose=False)


ana = analyze_random_walk("{}".format(path)+"QM9_histogram.pkl","{}".format(path)+"QM9_best_candidates.pkl", model=min_func)
X_2d = ana.comute_PCA()
#values = np.load("QM9_histogram_values.npz")["values"]

ana.evaluate_histogram(save_histogram=True)
pdb.set_trace()
#values 
values  = ana.values
#pdb.set_trace()
front, inds, mols = ana.compute_pareto_front()
ana.write_report(plotname)




print("Plot Pareto Front")
fig,ax1= plt.subplots(figsize=(8,8))
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
plt.ylabel("$E_{\\rm {gap}}$" + " [eV]", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)
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
plt.savefig("{}_pareto.pdf".format(plotname))
plt.close()

print("Plot Chemical Space PCA")

fig2,ax2= plt.subplots(figsize=(8,8))

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.spines['bottom'].set_position(('axes', -0.05))
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.spines['left'].set_position(('axes', -0.05))

sc = ax2.scatter(x=X_2d[:,0], y=X_2d[:,1],s=400,alpha=0.1,marker="o", c=ana.values[:,2],edgecolors='none')
ax2.set_xlabel("PC1", fontsize=21)
ax2.set_ylabel("PC2", fontsize=21,rotation=0, ha="left", y=1.05, labelpad=-50, weight=500)

clb = plt.colorbar(sc)
clb.set_label('Loss')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("{}_PCA.pdf".format(plotname))
plt.close()

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