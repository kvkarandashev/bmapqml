# Collection of routines that use rdkit.Chem.Draw for easy display of objects used throughout the chemxpl module.
from rdkit.Chem.Draw import rdMolDraw2D
from .rdkit_utils import chemgraph_to_rdkit, SMILES_to_egc
from .ext_graph_compound import ExtGraphCompound
from rdkit.Chem.rdmolops import RemoveHs
from copy import deepcopy
import numpy as np


class ChemGraphDrawing:
    def __init__(
        self,
        chemgraph=None,
        SMILES=None,
        size=(300, 300),
        bw_palette=True,
        kekulize=True,
        explicit_hydrogens=False,
        highlightAtoms=None,
        highlightAtomColor=None,
        highlightAtomColors=None,
        highlight_connecting_bonds=False,
        highlightBondTuples=None,
        highlightBondTupleColor=None,
        highlightBondTupleColors=None,
        highlightAtomRadius=None,
        highlightAtomRadii=None,
        highlightBondWidthMultiplier=None,
        bondLineWidth=None,
        baseFontSize=None,
    ):

        self.base_init(
            size=size,
            chemgraph=chemgraph,
            SMILES=SMILES,
            bw_palette=bw_palette,
            highlightBondWidthMultiplier=highlightBondWidthMultiplier,
            kekulize=kekulize,
            explicit_hydrogens=explicit_hydrogens,
            bondLineWidth=bondLineWidth,
            baseFontSize=baseFontSize,
        )
        self.highlight_init(
            highlightAtoms=highlightAtoms,
            highlightAtomColor=highlightAtomColor,
            highlightAtomColors=highlightAtomColors,
            highlight_connecting_bonds=highlight_connecting_bonds,
            highlightBondTuples=highlightBondTuples,
            highlightBondTupleColor=highlightBondTupleColor,
            highlightBondTupleColors=highlightBondTupleColors,
            highlightAtomRadius=highlightAtomRadius,
            highlightAtomRadii=highlightAtomRadii,
        )
        self.prepare_and_draw()

    def base_init(
        self,
        chemgraph=None,
        SMILES=None,
        bw_palette=True,
        highlightBondWidthMultiplier=None,
        size=(300, 300),
        resonance_struct_adj=None,
        kekulize=False,
        explicit_hydrogens=False,
        bondLineWidth=None,
        baseFontSize=None,
    ):
        if chemgraph is None:
            if SMILES is not None:
                self.chemgraph = SMILES_to_egc(SMILES).chemgraph
        else:
            self.chemgraph = chemgraph
        self.resonance_struct_adj = resonance_struct_adj

        self.explicit_hydrogens = explicit_hydrogens
        self.mol = chemgraph_to_rdkit(
            self.chemgraph,
            resonance_struct_adj=self.resonance_struct_adj,
            explicit_hydrogens=self.explicit_hydrogens,
        )

        self.kekulize = kekulize
        self.bw_palette = bw_palette
        self.drawing = rdMolDraw2D.MolDraw2DCairo(*size)

        do = self.drawing.drawOptions()
        if bw_palette:
            do.useBWAtomPalette()
        if highlightBondWidthMultiplier is not None:
            do.highlightBondWidthMultiplier = highlightBondWidthMultiplier
        if bondLineWidth is not None:
            do.bondLineWidth = bondLineWidth
        if baseFontSize is not None:
            do.baseFontSize = baseFontSize

    def highlight_init(
        self,
        highlightAtoms=None,
        highlightAtomColor=None,
        highlightAtomColors=None,
        highlight_connecting_bonds=False,
        highlightBondTuples=None,
        highlightBondTupleColor=None,
        highlightBondTupleColors=None,
        highlightAtomRadius=None,
        highlightAtomRadii=None,
    ):

        self.highlightAtoms = deepcopy(highlightAtoms)
        self.highlightAtomColors = deepcopy(highlightAtomColors)
        self.highlightAtomColor = highlightAtomColor

        self.highlightAtomRadius = highlightAtomRadius
        self.highlightAtomRadii = deepcopy(highlightAtomRadii)

        self.highlightBondTupleColor = highlightBondTupleColor
        self.highlightBondTupleColors = deepcopy(highlightBondTupleColors)
        self.highlight_connecting_bonds = highlight_connecting_bonds
        self.highlightBondTuples = deepcopy(highlightBondTuples)

        if self.highlightAtomColor is not None:
            self.highlight_atoms(self.highlightAtoms, self.highlightAtomColor)

        if self.highlightBondTupleColor:
            self.highlight_bonds(self.highlightBondTuples, self.highlightBondTupleColor)

        if self.highlight_connecting_bonds:
            self.highlight_bonds_connecting_atoms(
                self.highlightAtoms, self.highlightAtomColor
            )

    def highlight_atoms(self, atom_ids, highlight_color, wbonds=False, overwrite=False):
        if self.highlightAtomColors is None:
            self.highlightAtomColors = {}
        if self.highlightAtoms is None:
            self.highlightAtoms = []
        for ha in atom_ids:
            if (ha not in self.highlightAtomColors) or overwrite:
                self.highlightAtomColors[ha] = highlight_color
            if ha not in self.highlightAtoms:
                self.highlightAtoms.append(ha)
        if self.highlightAtomRadius is not None:
            if self.highlightAtomRadii is None:
                self.highlightAtomRadii = {}
            for ha in atom_ids:
                if (ha not in self.highlightAtomRadii) or overwrite:
                    self.highlightAtomRadii[ha] = self.highlightAtomRadius
        if wbonds:
            self.highlight_bonds_connecting_atoms(
                atom_ids, highlight_color, overwrite=overwrite
            )

    def highlight_bonds(self, bond_tuples, highlight_color, overwrite=False):
        self.check_highlightBondTupleColors()
        for bt in bond_tuples:
            if (bt not in self.highlightBondTupleColors) or overwrite:
                self.highlightBondTupleColors[bt] = highlight_color

    def highlight_bonds_connecting_atoms(
        self, atom_ids, highlight_color, overwrite=False
    ):
        connecting_bts = self.connecting_bond_tuples(atom_ids)
        self.highlight_bonds(connecting_bts, highlight_color, overwrite=overwrite)

    def check_highlightBondTupleColors(self):
        if self.highlightBondTupleColors is None:
            self.highlightBondTupleColors = {}

    def prepare_and_draw(self):
        if self.highlightBondTupleColors is not None:
            highlightBonds = []
            highlightBondColors = {}
            for bt in self.highlightBondTupleColors:
                bond_id = self.mol.GetBondBetweenAtoms(*bt).GetIdx()
                highlightBonds.append(bond_id)
                highlightBondColors[bond_id] = self.highlightBondTupleColors[bt]
        else:
            highlightBonds = None
            highlightBondColors = None

        rdMolDraw2D.PrepareAndDrawMolecule(
            self.drawing,
            self.mol,
            kekulize=self.kekulize,
            highlightAtomColors=self.highlightAtomColors,
            highlightAtoms=self.highlightAtoms,
            highlightBonds=highlightBonds,
            highlightBondColors=highlightBondColors,
            highlightAtomRadii=self.highlightAtomRadii,
        )

    def save(self, filename):
        self.drawing.WriteDrawingText(filename)

    def connecting_bond_tuples(self, atom_ids):
        output = []
        for atom_id in atom_ids:
            for neigh in self.chemgraph.neighbors(atom_id):
                if neigh < atom_id:
                    if neigh in atom_ids:
                        output.append((neigh, atom_id))
        return output


class FragmentPairDrawing(ChemGraphDrawing):
    def __init__(
        self,
        fragment_pair=None,
        bw_palette=True,
        size=(300, 300),
        resonance_struct_adj=None,
        highlight_fragment_colors=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        bondLineWidth=None,
        highlight_fragment_boundary=None,
        highlightAtomRadius=None,
        highlightBondWidthMultiplier=None,
        baseFontSize=None,
    ):
        # Initialize all basic quantities.
        self.base_init(
            chemgraph=fragment_pair.chemgraph,
            bw_palette=bw_palette,
            size=size,
            resonance_struct_adj=resonance_struct_adj,
            highlightBondWidthMultiplier=highlightBondWidthMultiplier,
            bondLineWidth=bondLineWidth,
            baseFontSize=baseFontSize,
        )
        # For starters only highlight the bonds connecting the two fragments.
        self.highlight_init(highlightAtomRadius=highlightAtomRadius)
        self.highlight_fragment_colors = highlight_fragment_colors
        self.highlight_fragment_boundary = highlight_fragment_boundary

        self.connection_tuples = []
        for tuples in fragment_pair.affected_status[0]["bonds"].values():
            self.connection_tuples += tuples

        if self.highlight_fragment_boundary is not None:
            self.highlight_bonds(
                self.connection_tuples, self.highlight_fragment_boundary
            )
        if self.highlight_fragment_colors is not None:
            for fragment_highlight, vertices in zip(
                self.highlight_fragment_colors, fragment_pair.sorted_vertices
            ):
                self.highlight_atoms(vertices, fragment_highlight, wbonds=True)
        self.prepare_and_draw()


def draw_chemgraph_to_file(cg, filename, **kwargs):
    """
    Draw a chemgraph in a PNG file.
    """
    cgd = ChemGraphDrawing(chemgraph=cg, **kwargs)
    cgd.save(filename)


def draw_fragment_pair_to_file(fragment_pair, filename, **kwargs):
    """
    Draw a fragment pair in a PNG file.
    """
    fpd = FragmentPairDrawing(fragment_pair=fragment_pair, **kwargs)
    fpd.save(filename)
