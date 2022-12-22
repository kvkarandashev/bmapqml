# Merge several histrogram files with a final restart file creating a single restart.
from bmapqml.chemxpl.random_walk import RandomWalk
from bmapqml.utils import loadpkl, ispklfile
import sys, subprocess

output_restart = sys.argv[1]

input_restart = sys.argv[2]

histogram_dumps = sys.argv[3:]

if len(histogram_dumps) == 0:  # No changes to input file needed.
    subprocess.run(["cp", input_restart, output_restart])

compress_restart = not ispklfile(input_restart)

temp_rw = RandomWalk(
    compress_restart=compress_restart,
    restart_file=input_restart,
    keep_histogram=True,
    keep_full_trajectory=True,
)

temp_rw.restart_from()

for histogram_dump in histogram_dumps:
    added_histogram = loadpkl(histogram_dump)
    temp_rw.merge_histogram(added_histogram)

temp_rw.make_restart(
    restart_file=output_restart, tarball=(not ispklfile(output_restart))
)
