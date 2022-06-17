
from bmapqml.utils import analyze_random_walk
import numpy as np
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj,Rdkit_properties
import pdb

def compute_pareto_front(values):
    
    """
    values: array of function values
    Returns indices of points in the pareto front.
    pareto_front as well as the indices
    """

    Xs, Ys = values[:,0], values[:,1]
    maxY = False
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]

    #return pareto_front
    
    
    
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)


    inds = []   
    for pair in pareto_front:
        if pair[0] in values[:,0] and pair[1] in values[:,1]:
            inds.append( int(np.where(values[:,0]==pair[0]) and np.where(values[:,1]==pair[1])[0][0]))
    
    inds =np.array(inds)
    inds  = inds.astype('int')
    #
    #print(2345)
    #pareto_front = np.array(pareto_front)

    return np.array(pareto_front), np.int_(inds)


path = "/home/jan/projects/MOLOPT/do_sim/bias/both/9/"


WEIGHTS = np.array([ (1/1.9), (1/6.8)])

min_func = multi_obj(
    [QM9_properties("/store/common/jan/qm9/KRR_12000_atomization",verbose=False),QM9_properties("/store/common/jan/qm9/KRR_12000_gap",verbose=False)
], WEIGHTS, verbose=True)


ana = analyze_random_walk("{}".format(path)+"QM9_histogram.pkl","{}".format(path)+"QM9_best_candidates.pkl", model=min_func)



#values = ana.evaluate_histogram()

values = np.load("QM9_histogram_values.npz")["values"]
ana.values = values

pdb.set_trace()
front, inds, mols = ana.compute_pareto_front()
# compute_pareto_front(values[:,:2])
#mols = np.array(ana.histogram)
#mols[inds]
#'O=CC1=NC2C=CC=C12' in mols[inds.astype(int)]


import matplotlib.pyplot as plt
import matplotlib.tri as tri
fig,ax1= plt.subplots(figsize=(8,8))


sub = 100
p1_acc  = values[:,0][::sub]
p2_acc  = values[:,1][::sub]
summe = values[:,2][::sub]
#summe  = accepted[" dGANDen"].values


#All_dG_new = All_dG_new*0.0175
xi = np.linspace(min(p1_acc), max(p1_acc), 100)
yi = np.linspace(min(p2_acc), max(p2_acc), 100)

# Perform linear interpolation of the data (x,y)
# on a grid defined by (xi,yi)
triang = tri.Triangulation(p1_acc, p2_acc)
interpolator = tri.LinearTriInterpolator(triang,summe)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

ax1.contour(xi, yi, zi, levels=18, linewidths=0.5, colors='k')
#ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
sc = ax1.scatter(p1_acc, p2_acc,s =0.5, c=summe)
plt.xlabel("$E_{\\rm {at}}/ N^{\\rm tot}$"  + " [eV]", fontsize=21)
plt.ylabel("$E_{\\rm {gap}}$" + " [eV]", fontsize=21)

clb = plt.colorbar(sc)
#clb.set_label("Step"+" [eV]") 
clb.set_label("Loss") 

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['bottom'].set_position(('axes', -0.05))
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['left'].set_position(('axes', -0.05))

plt.plot(front[:,0],front[:,1],'*', color='red')


plt.plot(front[:,0],front[:,1],'r--',linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("sum_const.pdf")




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

