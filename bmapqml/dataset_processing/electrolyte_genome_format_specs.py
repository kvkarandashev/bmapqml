try:
    from ..orb_ml.oml_compound import OML_Slater_pair, OML_compound
except:
    print("Failed to import bmapqml.orb_ml module; quantity estimates unavailable.")

from ..utils import write_xyz_file
import os


class QuantitiesNotAvailableError(Exception):
    pass


class TaskIDRepeated(Exception):
    pass


from urllib.parse import quote_plus
from ..data import conversion_coefficient

def_float_format = "{:.8E}"

au_to_eV_mult = conversion_coefficient["au_eV"]


def Ionization_Energy(xyz_name, calc_type="UHF", **oml_comp_kwargs):
    return electron_energy_change(xyz_name, 1, calc_type=calc_type, **oml_comp_kwargs)


def Electron_Affinity(xyz_name, calc_type="UHF", **oml_comp_kwargs):
    return -electron_energy_change(xyz_name, -1, calc_type=calc_type, **oml_comp_kwargs)


def electron_energy_change(
    xyz_name,
    charge_change,
    calc_type="UHF",
    charge=0,
    localization_procedure=None,
    **oml_comp_kwargs
):
    new_charge = charge + charge_change
    Slater_pair = OML_Slater_pair(
        xyz=xyz_name,
        mats_savefile=xyz_name,
        calc_type=calc_type,
        second_oml_comp_kwargs={"charge": new_charge},
        localization_procedure=localization_procedure,
        **oml_comp_kwargs
    )
    Slater_pair.run_calcs()
    return (Slater_pair.comps[1].e_tot - Slater_pair.comps[0].e_tot) * au_to_eV_mult


quant_properties = {
    "IE": ("eV", Ionization_Energy),
    "EA": ("eV", Electron_Affinity),
}

quant_data_type = {
    "smiles": str,
    "charge": int,
    "task_id": str,
    "IE": float,
    "EA": float,
}


def quant_converter(name):
    if name in quant_data_type:
        return quant_data_type[name]
    else:
        return float


def convert_quant(s, name):
    return quant_converter(name)(s)


def extract_quant_from_xyz(xyz_file, name):
    file = open(xyz_file, "r")
    lines = file.readlines()
    file.close()
    for chunk in lines[1].split():
        csplit = chunk.split(":")
        if csplit[0] == name:
            return convert_quant(csplit[1], name)
    return None


class Quantity:
    def __init__(self, quant_name):
        self.name = quant_name
        quant_tuple = quant_properties[quant_name]
        self.dimensionality = quant_tuple[0]
        self.estimate_function = quant_tuple[1]

    def extract_xyz(self, filename):
        return extract_quant_from_xyz(filename, self.name)

    def OML_calc_quant(self, xyz_name, calc_type="UHF", **oml_comp_kwargs):
        charge = extract_quant_from_xyz(xyz_name, "charge")

        return quant_properties[self.name][1](
            xyz_name, calc_type=calc_type, charge=charge, **oml_comp_kwargs
        )


#   For importing from materials project website.
def MAPI_KEY():
    try:
        return os.environ["MAPI_KEY"]
    except LookupError:
        print("MAPI_KEY environmental variable needs to be set.")
        quit()


saved_data_fields = ["smiles", "charge", "IE", "EA", "task_id"]


def download_all_xyzs(dump_dir="."):  # , pkl_dump="mp_api_moldump.pkl"):
    from mp_api.client import MPRester
    from ..utils import mkdir  # , dump2pkl, loadpkl

    # Done in order to not spam the API too much if something keeps going wrong.
    # Does not work because mol_docs are not pickle-able.
    #    if os.path.isfile(pkl_dump):
    #        mol_docs = loadpkl(pkl_dump)
    #    else:
    mpr = MPRester(MAPI_KEY())
    mol_docs = mpr.molecules.search()
    #        dump2pkl(mol_docs, pkl_dump)

    mkdir(dump_dir)

    for mol_doc in mol_docs:
        mol_doc_dict = mol_doc.__dict__
        mol = mol_doc.molecule

        extra_string = ""

        for field in saved_data_fields:
            if field in mol_doc_dict:
                extra_string += field + ":" + str(mol_doc_dict[field]) + " "
        extra_string = extra_string[:-1]

        output_filename = dump_dir + "/" + mol_doc_dict["task_id"] + ".xyz"

        if hasattr(mol, "cart_coords"):
            coords = mol.cart_coords
            elements = [None for _ in range(len(coords))]
            for symbol in mol.symbol_set:
                indices = mol.indices_from_symbol(symbol)
                for i in indices:
                    elements[i] = symbol
            if None in elements:
                raise Exception()
        else:
            coords = []
            elements = []

        write_xyz_file(
            coords,
            output_filename,
            elements=elements,
            extra_string=extra_string,
        )
