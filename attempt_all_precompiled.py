# An utility that execute all "precompiled" statements in the code.
import glob, os
from bmapqml.precompilation import precompiled

# TODO if needed add kwargs
def attempt_all_precompiled(
    parent_module_name="bmapqml", file_in_module_root_dir=__file__
):
    source_dir = os.path.dirname(file_in_module_root_dir) + "/" + parent_module_name
    needed = []
    for sd in os.walk(source_dir):
        for f in sd[2]:
            if f[-3:] == ".py":
                r = open(sd[0] + "/" + f, "r")
                for line in r.readlines():
                    if (
                        line.split("(")[0].strip() == "precompiled"
                        and line.split(")")[-1].strip() == ""
                    ):
                        args = line.split("(")[1].split(")")[-2].split(",")
                        for arg in args:
                            if "=" in arg:
                                break
                            else:
                                needed.append(arg.strip()[1:-1])
                r.close()
    precompiled(*needed)


if __name__ == "__main__":
    attempt_all_precompiled()
