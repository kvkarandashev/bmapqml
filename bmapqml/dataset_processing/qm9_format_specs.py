# IMPORTANT: Note that for the QM9 total potential energy results what was actually predicted was internal energy at 0K with
# total potential energy used as baseline in the Delta-ML scheme.

from ..orb_ml.oml_compound import OML_compound
from ..utils import checked_input_readlines

try:
    from ..chemxpl.utils import SMILES_to_egc, xyz2mol_extgraph, RdKitFailure
    from ..chemxpl.valence_treatment import InvalidAdjMat
except:
    pass


def HOMO_en(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc='', **other_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc, **other_kwargs)
    oml_comp.run_calcs()
    return oml_comp.HOMO_en()
    
def LUMO_en(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc='', **other_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc, **other_kwargs)
    oml_comp.run_calcs()
    return oml_comp.LUMO_en()

def HOMO_LUMO_gap(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc='', **other_kwargs):
    return LUMO_en(xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc, **other_kwargs)-HOMO_en(xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc, **other_kwargs)

def potential_energy(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)
    oml_comp.run_calcs()
    return oml_comp.e_tot

quant_properties = {'Dipole moment' : (6, 'Debye'),
                'Isotropic polarizability' : (7, 'Bohr^3'),
                'HOMO eigenvalue': (8, 'Hartree', HOMO_en),
                'LUMO eigenvalue': (9, 'Hartree', LUMO_en),
                'HOMO-LUMO gap': (10, 'Hartree', HOMO_LUMO_gap),
                'Electronic spacial extent': (11, 'Bohr^2'),
                'Zero point vibrational energy': (12, 'Hartree'),
                'Internal energy at 0 K': (13, 'Hartree', potential_energy),
                'Internal energy at 298.15 K': (14, 'Hartree'),
                'Enthalpy at 298.15 K': (15, 'Hartree'),
                'Free energy at 298.15 K': (16, 'Hartree'),
                'Heat capacity at 298.15 K': (17, 'cal/(mol K)'),
                'Highest vibrational frequency': (18, 'cm^-1')}

class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        self.qm9_id=quant_properties[quant_name][0]
        self.dimensionality=quant_properties[quant_name][1]
    def extract_xyz(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        if self.name == 'Highest vibrational frequency':
            first_line_passing=True
        for l in lines:
            lsplit=l.split()
            if self.name == 'Highest vibrational frequency':
                if first_line_passing:
                    first_line_passing=False
                    continue
                try:
                    output=max([float(freq_str) for freq_str in lsplit]) # this will fail for all lines but for the one with molecule number (first line) and frequencies
                    break
                except:
                    continue
            else:
                if lsplit[0] == "gdb":
                    output=float(lsplit[self.qm9_id-1])
                    break
        file.close()
        return output
    def extract_byprod_result(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        for l in lines:
            lsplit=l.split()
            if int(lsplit[0]) == self.qm9_id:
                output=float(lsplit[1])
                break
        file.close()
        return output
    def OML_calc_quant(self, xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc='', **other_kwargs):
        return quant_properties[self.name][2](xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc, **other_kwargs)
    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.qm9_id)+" "+str(val)+"\n")

def read_str_rep(xyz_input, offset):
    lines=checked_input_readlines(xyz_input)
    natoms=int(lines[0])
    str_rep_line_id=natoms+offset
    return lines[str_rep_line_id].split()[0]

def read_SMILES(xyz_input):
    return read_str_rep(xyz_input, 3)

def read_InChI(xyz_input):
    return read_str_rep(xyz_input, 4)


def xyz_SMILES_consistent(xyz_file):
    SMILES=read_SMILES(xyz_file)
    try:
        egc1=SMILES_to_egc(SMILES)
    except InvalidAdjMat:
        return False
    except RdKitFailure:
        return False
    try:
        egc2=xyz2mol_extgraph(xyz_file)
    except:
        return False
    return egc1==egc2
