#l is the seed of the simulation
#mol is the target molecule
#d is the distance in representation space
#thickness is the thickness of the shell
# i.e. here we sample interval dR = [5,8]
# Nsteps # of MC steps

rm out*
l=1
mol=asperin
d=8
python chemspacesampler.py -Nsteps 100 -dmin ${d} -label ${l} -thickness 3.0 >> out_${d}_${l}