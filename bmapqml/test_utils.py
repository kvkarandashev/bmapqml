# Miscellaneous functions and classes used for testing convenience.
import random, glob, datetime

def_float_format = "{:.8E}"


class logfile:
    def __init__(self, base_name=None, ending=None):
        self.not_empty = base_name is not None
        if self.not_empty:
            full_filename = base_name
            if ending is not None:
                full_filename += ending
            self.output = open(full_filename, "w")

    def write(self, *args):
        if self.not_empty:
            print(*args, file=self.output)

    def close(self):
        if self.not_empty:
            self.output.close()

    def export_quantity_array(self, quantity_array):
        if self.not_empty:
            for quant_val in quantity_array:
                self.output.write((def_float_format + "\n").format(quant_val))

    def export_matrix(self, matrix_in):
        if self.not_empty:
            for id1, row in enumerate(matrix_in):
                for id2, val in enumerate(row):
                    self.output.write(
                        ("{} {} " + def_float_format + "\n").format(id1, id2, val)
                    )

    def randomized_export_3D_arr(self, mat_wders, seed):
        random_generator = random.Random(seed)
        format_string = "{} {} "
        for _ in range(3):
            format_string += " " + def_float_format
        format_string += "\n"
        if self.not_empty:
            for id1, row in enumerate(mat_wders):
                for id2, kern_els in enumerate(row):
                    self.output.write(
                        format_string.format(
                            id1,
                            id2,
                            kern_els[0],
                            kern_els[1],
                            random_generator.sample(list(kern_els[2:]), 1)[0],
                        )
                    )


def dirs_xyz_list(xyz_dir):
    output = glob.glob(xyz_dir + "/*.xyz")
    output.sort()
    return output


def timestamp(title=None, start_time=None):
    """
    Prints current time along with another string.
    title : string to be printed close to the timestamp, empty by default.
    """
    cur_time = datetime.datetime.now()
    if start_time is None:
        displayed_time = cur_time
    else:
        displayed_time = cur_time - start_time
    if title is not None:
        print(title, displayed_time)
    return cur_time
