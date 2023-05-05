try:
    import tblite.interface as tb
except:
    pass
from .aux_classes import PseudoMF, PseudoMole
import numpy as np

# Extracts atom AO ranges and angular momenta from a tblite calculator object.
def tblite_ao_ranges_angular_momenta(tblite_calc):
    ao_map = tblite_calc.get("orbital-map")
    sh_map = tblite_calc.get("shell-map")
    sh_am = tblite_calc.get("angular-momenta")
    ao_ranges = []
    angular_momenta = np.zeros((len(ao_map),), dtype=int)
    prev_range_start = 0
    prev_atom = 0
    for ao_id, sh_id in enumerate(ao_map):
        angular_momenta[ao_id] = sh_am[sh_id]
        if sh_map[sh_id] != prev_atom:
            ao_ranges.append((prev_range_start, ao_id))
            prev_range_start = ao_id
            prev_atom = sh_map[sh_id]
    ao_ranges.append((ao_id, len(ao_map)))
    return ao_ranges, angular_momenta


def generate_pyscf_mf_mol(oml_compound):
    calculator = tb.Calculator(
        oml_compound.calc_type, oml_compound.nuclear_charges, oml_compound.coordinates
    )

    ao_ranges, angular_momenta = tblite_ao_ranges_angular_momenta(calculator)

    calculator.set("save-integrals", 1)

    res = calculator.singlepoint()

    res_dict = res.dict()

    pMF = PseudoMF(
        e_tot=res_dict["energy"],
        mo_coeff=res_dict["orbital-coefficients"].T,
        mo_occ=res_dict["orbital-occupations"],
        mo_energy=res_dict["orbital-energies"],
    )

    pMole = PseudoMole(res_dict["overlap-matrix"], ao_ranges, angular_momenta)
    return pMF, pMole
