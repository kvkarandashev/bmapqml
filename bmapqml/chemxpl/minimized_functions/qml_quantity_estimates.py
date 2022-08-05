from qml.representations import generate_slatm
from .ml_quantity_estimates import FF_based_model


class SLATM_FF_based_model(FF_based_model):
    """
    An inheritor to FF_based_model that uses SLATM representation.
    """

    def __init__(self, *args, mbtypes=None, **other_kwargs):

        super().__init__(*args, **other_kwargs)
        self.mbtypes = mbtypes
        self.rep_func = generate_slatm

    def verify_mbtypes(self, xyz_files):
        from qml.representations import get_slatm_mbtypes

        all_nuclear_charges = []
        max_natoms = 0
        for f in xyz_files:
            (
                nuclear_charges,
                _,
                _,
                _,
            ) = read_xyz_file(f)
            max_natoms = max(len(nuclear_charges), max_natoms)
            all_nuclear_charges.append(nuclear_charges)
        np_all_nuclear_charges = np.zeros(
            (len(all_nuclear_charges), max_natoms), dtype=int
        )
        for mol_id, nuclear_charges in enumerate(all_nuclear_charges):
            cur_natoms = len(nuclear_charges)
            np_all_nuclear_charges[mol_id, :cur_natoms] = nuclear_charges[:]
        self.mbtypes = get_slatm_mbtypes(np_all_nuclear_charges)

    def coord_representation_func(self, coordinates, nuclear_charges):
        return self.rep_func(coordinates, nuclear_charges, self.mbtypes)
