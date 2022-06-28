import glob, sys, os
from bmapqml.utils import mkdir
from bmapqml.chemxpl.utils import xyz2mol_extgraph, egc_with_coords, MMFFInconsistent, write_egc2xyz

if len(sys.argv)==1:
    print("Use location of xyz directory as the script's argument.")
    quit()
else:
    xyz_dir=sys.argv[1]
    new_dirname=os.getcwd()+"/MMFF_"+os.path.basename(xyz_dir)

xyz_list=glob.glob(xyz_dir+"/*.xyz")
xyz_list.sort()

mkdir(new_dirname)

bad_xyzs=[]

for xyz in xyz_list:
    print(xyz)
    cur_egc=xyz2mol_extgraph(xyz)
    try:
        new_egc=egc_with_coords(cur_egc)
    except MMFFInconsistent:
        new_egc=None
    if new_egc is None:
        bad_xyzs.append(xyz)
    else:
        new_xyz=new_dirname+"/"+os.path.basename(xyz)
        write_egc2xyz(new_egc, new_xyz)
    
print("BAD XYZ FILES:")
for xyz in bad_xyzs:
    print(xyz)
