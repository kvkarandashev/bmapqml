# An utility script clearing molpro directory of all compiled sources
import os, subprocess

# Delete all __pycache__folders.
molopt_source_dir=os.path.dirname(__file__)+"/molopt"
checked_dirs=[molopt_source_dir]
while (len(checked_dirs)!=0):
    checked_item=checked_dirs[0]
    new_items=os.listdir(checked_item)
    for ni in new_items:
        true_ni=checked_item+"/"+ni
        if ni == "__pycache__":
            subprocess.run(["rm", "-Rf", true_ni])
        else:
            if os.path.isdir(true_ni):
                checked_dirs.append(true_ni)
            else:
                if ni.split('.')[-2:]==[ "cpython-39-x86_64-linux-gnu", "so" ]:
                    subprocess.run(["rm", "-f", ni])
    del(checked_dirs[0])
