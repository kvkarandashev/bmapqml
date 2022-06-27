from bmapqml.dataset_processing.qm9_format_specs import read_SMILES, read_InChI
from bmapqml.chemxpl.utils import SMILES_to_egc, InChI_to_egc, xyz2mol_extgraph

example_xyz_files=["dsgdb9nsd_000001.xyz", "dsgdb9nsd_000230.xyz"]

for xyz_file in example_xyz_files:
    SMILES=read_SMILES(xyz_file)
    InChI=read_InChI(xyz_file)
    print("Molecule:", SMILES, InChI)
    egc1=SMILES_to_egc(SMILES)
    egc2=InChI_to_egc(InChI)
    egc3=xyz2mol_extgraph(xyz_file)
    print("Matches:", egc1==egc2, egc2==egc3)
