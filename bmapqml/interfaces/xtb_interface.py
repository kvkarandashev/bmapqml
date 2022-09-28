import subprocess, os
from ..utils import (
    rmdir,
    mktmpdir,
    write_xyz_file,
    mkdir,
    write2file,
    OptionUnavailableError,
)

quant_name_lists = {
    "total_energy": ["total", "energy"],
    "solvation_energy": ["->", "Gsolv"],
    "gap": ["HOMO-LUMO", "gap"],
}


def xtb_output_extract_quants(xtb_output, quant_names):
    lsplits = [l.split() for l in xtb_output.split("\n")]
    output = {}
    for quant_name in quant_names:
        name_list = quant_name_lists[quant_name]
        for lsplit in lsplits:
            if len(lsplit) > 3:
                if (
                    (lsplit[0] == "::")
                    and (lsplit[-1] == "::")
                    and (lsplit[1:-3] == name_list)
                ):
                    output[quant_name] = float(lsplit[-3])
                    break
        if quant_name not in output:
            raise Exception
    return output


def xTB_results(
    coordinates,
    elements=None,
    nuclear_charges=None,
    charge=0,
    spin=0,
    solvent=None,
    workdir=None,
    quantities=["total_energy"],
    geom_opt=None,
):
    mydir = os.getcwd()

    for quant in quantities:
        if quant not in quant_name_lists:
            raise OptionUnavailableError
    if workdir is None:
        true_workdir = mktmpdir()
    else:
        mkdir(workdir)
        true_workdir = workdir
    xyz_name = "xtb_inp.xyz"
    os.chdir(true_workdir)
    write_xyz_file(
        coordinates, xyz_name, elements=elements, nuclear_charges=nuclear_charges
    )

    cmd_arguments = ["xtb", xyz_name]

    if charge != 0:
        cmd_arguments += ["--chrg", str(charge)]
    if solvent is not None:
        cmd_arguments += ["--alpb", solvent]
    if spin != 0:
        cmd_arguments += ["--uhf", str(spin)]

    if geom_opt is not None:
        subprocess.run(cmd_arguments + ["--opt", geom_opt], capture_output=True)
        cmd_arguments[1] = "xtbopt.xyz"

    xtb_output = subprocess.run(cmd_arguments, capture_output=True)
    stdout = xtb_output.stdout.decode("utf-8")

    output = xtb_output_extract_quants(stdout, quantities)

    if workdir is not None:
        # Save the stdout and stderr files.
        # TO-DO dedicated options for writing.
        write2file(stdout, "xtb.stdout")
        write2file(xtb_output.stderr.decode("utf-8"), "xtb.stderr")

    os.chdir(mydir)

    if workdir is None:
        rmdir(true_workdir)
    return output
