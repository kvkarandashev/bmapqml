# Collection of routines that use rdkit.Chem.Draw for easy display of objects used throughout the chemxpl module.
from rdkit.Chem.Draw import rdMolDraw2D
from .rdkit_utils import chemgraph_to_rdkit, SMILES_to_egc
from copy import deepcopy
import numpy as np
from .valence_treatment import ChemGraph, sorted_tuple
from .modify import FragmentPair
import itertools
from .modify import (
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    valence_change_add_atoms_possibilities,
    valence_change_remove_atoms_possibilities,
    change_bond_order_valence,
)
from rdkit.Chem import rdAbbreviations
from .random_walk import TrajectoryPoint, full_change_list
from .periodic import element_name

# Some colors that I think look good on print.
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)

LIGHTRED = (1.0, 0.5, 0.5)
LIGHTGREEN = (0.5, 1.0, 0.5)
LIGHTBLUE = (0.5, 0.5, 1.0)


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
        resonance_struct_adj=None,
        abbrevs=None,
        abbreviate_max_coverage=1.0,
    ):
        """
        Create an RdKit illustration depicting a partially highlighted ChemGraph.
        """
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
            resonance_struct_adj=resonance_struct_adj,
            highlightAtoms=highlightAtoms,
            highlightAtomColor=highlightAtomColor,
            highlightAtomColors=highlightAtomColors,
            highlight_connecting_bonds=highlight_connecting_bonds,
            highlightBondTuples=highlightBondTuples,
            highlightBondTupleColor=highlightBondTupleColor,
            highlightBondTupleColors=highlightBondTupleColors,
            highlightAtomRadius=highlightAtomRadius,
            highlightAtomRadii=highlightAtomRadii,
            abbrevs=abbrevs,
            abbreviate_max_coverage=abbreviate_max_coverage,
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
        highlightAtoms=None,
        highlightAtomColor=None,
        highlightAtomColors=None,
        highlight_connecting_bonds=False,
        highlightBondTuples=None,
        highlightBondTupleColor=None,
        highlightBondTupleColors=None,
        highlightAtomRadius=None,
        highlightAtomRadii=None,
        abbrevs=None,
        abbreviate_max_coverage=1.0,
    ):
        if chemgraph is None:
            if SMILES is not None:
                self.chemgraph = SMILES_to_egc(SMILES).chemgraph
        else:
            self.chemgraph = chemgraph
        self.resonance_struct_adj = resonance_struct_adj

        self.explicit_hydrogens = explicit_hydrogens

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

        self.highlightAtoms = deepcopy(highlightAtoms)
        self.highlightAtomColors = deepcopy(highlightAtomColors)
        self.highlightAtomColor = highlightAtomColor

        self.highlightAtomRadius = highlightAtomRadius
        self.highlightAtomRadii = deepcopy(highlightAtomRadii)

        self.highlightBondTupleColor = highlightBondTupleColor
        self.highlightBondTupleColors = deepcopy(highlightBondTupleColors)
        self.highlight_connecting_bonds = highlight_connecting_bonds
        self.highlightBondTuples = deepcopy(highlightBondTuples)

        self.abbrevs = abbrevs
        self.abbreviate_max_coverage = abbreviate_max_coverage

        if self.highlightAtomColor is not None:
            self.highlight_atoms(self.highlightAtoms, self.highlightAtomColor)

        if self.highlightBondTupleColor:
            self.highlight_bonds(self.highlightBondTuples, self.highlightBondTupleColor)

        if self.highlight_connecting_bonds:
            self.highlight_bonds_connecting_atoms(
                self.highlightAtoms, self.highlightAtomColor
            )

    def highlight_atoms(self, atom_ids, highlight_color, wbonds=False, overwrite=False):
        if highlight_color is None:
            return
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
        if highlight_color is None:
            return
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
        self.mol = chemgraph_to_rdkit(
            self.chemgraph,
            resonance_struct_adj=self.resonance_struct_adj,
            explicit_hydrogens=self.explicit_hydrogens,
            extra_valence_hydrogens=True,
        )
        if self.abbrevs is not None:
            used_abbrevs = rdAbbreviations.GetDefaultAbbreviations()
            self.full_mol = self.mol
            self.mol = rdAbbreviations.CondenseMolAbbreviations(
                self.full_mol, used_abbrevs, maxCoverage=self.abbreviate_max_coverage
            )

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
        abbrevs=None,
        abbreviate_max_coverage=1.0,
    ):
        """
        Create an RdKit illustration depicting a FragmentPair with atoms and bonds highlighted according to membership.
        """
        # Initialize all basic quantities.
        self.base_init(
            chemgraph=fragment_pair.chemgraph,
            bw_palette=bw_palette,
            size=size,
            resonance_struct_adj=resonance_struct_adj,
            highlightBondWidthMultiplier=highlightBondWidthMultiplier,
            bondLineWidth=bondLineWidth,
            baseFontSize=baseFontSize,
            abbrevs=abbrevs,
            abbreviate_max_coverage=abbreviate_max_coverage,
            highlightAtomRadius=highlightAtomRadius,
        )
        # For starters only highlight the bonds connecting the two fragments.
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
                if fragment_highlight is not None:
                    self.highlight_atoms(vertices, fragment_highlight, wbonds=True)
        self.prepare_and_draw()


def ObjDrawing(obj, **kwargs):
    if isinstance(obj, ChemGraph):
        return ChemGraphDrawing(obj, **kwargs)
    if isinstance(obj, FragmentPair):
        return FragmentPairDrawing(obj, **kwargs)


bond_changes = [change_bond_order, change_bond_order_valence]

valence_changes = [
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
]


class ModificationPathIllustration(ChemGraphDrawing):
    def __init__(
        self,
        cg,
        modification_path,
        change_function,
        color_change=None,
        color_change_neighbors=None,
        **other_image_params
    ):
        """
        Illustrate a modification path with simple moves as applied to a ChemGraph object.
        """
        self.base_init(chemgraph=cg, **other_image_params)
        self.modification_path = modification_path
        self.change_function = change_function
        self.color_change = color_change
        self.color_change_neighbors = color_change_neighbors
        self.highlight_atoms(self.change_neighbor_atoms(), self.color_change_neighbors)
        self.highlight_atoms(self.change_atoms(), self.color_change, wbonds=True)
        self.init_resonance_struct_adj()
        self.prepare_and_draw()

    def init_resonance_struct_adj(self):
        if self.chemgraph.resonance_structure_map is None:
            return
        affected_resonance_region = None
        changed_atom = None
        if self.change_function in bond_changes:
            st = sorted_tuple(*self.change_atoms())
            if st in self.chemgraph.resonance_structure_map:
                affected_resonance_region = self.chemgraph.resonance_structure_map[st]
                res_struct_id = self.modification_path[1][-1]

        if self.change_function in [replace_heavy_atom, change_valence]:
            changed_atom = self.change_atoms()[0]

        if self.change_function == replace_heavy_atom:
            res_struct_id = self.modification_path[1][1]

        if self.change_function == change_valence:
            res_struct_id = self.modification_path[1][1]

        if changed_atom is not None:
            for i, extra_valence_ids in enumerate(
                self.chemgraph.resonance_structure_inverse_map
            ):
                if changed_atom in extra_valence_ids:
                    affected_resonance_region = i

        if (affected_resonance_region is not None) and (res_struct_id is not None):
            self.resonance_struct_adj = {affected_resonance_region: res_struct_id}

    def change_atoms(self):
        if self.change_function == remove_heavy_atom:
            orig_atoms = [self.modification_path[1][0]]
        if self.change_function == change_valence_remove_atoms:
            orig_atoms = self.modification_path[2][0]
        if self.change_function in [remove_heavy_atom, change_valence_remove_atoms]:
            neigh_atom = self.chemgraph.neighbors(orig_atoms[0])[0]
            return orig_atoms + [neigh_atom]

        if self.change_function in [add_heavy_atom_chain, change_valence_add_atoms]:
            return []

        if self.change_function in bond_changes:
            return list(self.modification_path[1][:2])

        if self.change_function == replace_heavy_atom:
            return [self.modification_path[1][0]]

        if self.change_function == change_valence:
            return [self.modification_path[0]]

        raise Exception()

    def change_neighbor_atoms(self):
        if self.change_function in bond_changes:
            return self.change_atoms()

        if self.change_function == remove_heavy_atom:
            return [self.change_atoms()[1]]

        if self.change_function in [add_heavy_atom_chain, change_valence_add_atoms]:
            return [self.modification_path[1]]

        if self.change_function in [change_valence, replace_heavy_atom]:
            return []

        raise Exception()


# class BeforeAfterIllustration:
#    def __init__(self, cg, modification_path, change_function, **other_image_params):
#        """
#        Create a pair of illustrations corresponding to a modification_path.
#        """
#        from .random_walk import egc_change_func
#
#        new_cg = egc_change_func(cg, modification_path, change_function)


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


def all_possible_resonance_struct_adj(obj):
    """
    All values of resonance_struct_adj dictionnary appearing in *Drawing objects that are valid for a given object.
    """
    if isinstance(obj, ChemGraph):
        cg = obj
    else:
        cg = obj.chemgraph
    iterators = [
        list(range(len(res_struct_orders)))
        for res_struct_orders in cg.resonance_structure_orders
    ]
    if len(iterators) == 0:
        return [None]
    output = []
    for res_adj_ids in itertools.product(*iterators):
        new_dict = {}
        for reg_id, adj_id in enumerate(res_adj_ids):
            new_dict[reg_id] = adj_id
        output.append(new_dict)
    return output


def draw_all_possible_resonance_structures(
    obj, filename_prefix, filename_suffix=".png", **kwargs
):
    """
    Draw variants of an object with all possible resonance structures.
    """
    for rsa_id, resonance_struct_adj in enumerate(
        all_possible_resonance_struct_adj(obj)
    ):
        cur_drawing = ObjDrawing(
            obj, resonance_struct_adj=resonance_struct_adj, **kwargs
        )
        cur_drawing.save(filename_prefix + str(rsa_id) + filename_suffix)


def first_mod_path(tp):
    output = []
    subd = tp.modified_possibility_dict
    while isinstance(subd, list) or isinstance(subd, dict):
        if isinstance(subd, list):
            output.append(subd[0])
            subd = None
        if isinstance(subd, dict):
            new_key = list(subd.keys())[0]
            subd = subd[new_key]
            output.append(new_key)
    return output


default_randomized_change_params = {
    "change_prob_dict": full_change_list,
    "possible_elements": ["C"],
    "added_bond_orders": [1],
    "chain_addition_tuple_possibilities": False,
    "bond_order_changes": [-1, 1],
    "bond_order_valence_changes": [-2, 2],
    "max_fragment_num": 1,
    "added_bond_orders_val_change": [1, 2],
}


def draw_all_modification_possibilities(
    cg,
    filename_prefix,
    filename_suffix=".png",
    randomized_change_params=default_randomized_change_params,
    **kwargs
):
    cur_tp = TrajectoryPoint(cg=cg)
    # Check that cg satisfies the randomized_change_params_dict
    randomized_change_params = deepcopy(randomized_change_params)

    for ha in cg.hatoms:
        el = element_name[ha.ncharge]
        if el not in randomized_change_params["possible_elements"]:
            randomized_change_params["possible_elements"].append(el)

    cur_tp.init_possibility_info(**randomized_change_params)
    cur_tp.modified_possibility_dict = cur_tp.possibility_dict
    counter = 0
    while cur_tp.modified_possibility_dict != {}:
        full_mod_path = first_mod_path(cur_tp)
        counter += 1
        mpi = ModificationPathIllustration(
            cg, full_mod_path[1:], full_mod_path[0], **kwargs
        )
        cur_tp.delete_mod_path(full_mod_path)
        mpi.save(filename_prefix + str(counter) + filename_suffix)
