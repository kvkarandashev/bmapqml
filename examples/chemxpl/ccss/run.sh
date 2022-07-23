rm out*
l=1
mol=asperin
d=7
python chemspacesampler.py -Nsteps 200 -dmin ${d} -label ${l} -thickness 2.0 >> out_${d}_${l}