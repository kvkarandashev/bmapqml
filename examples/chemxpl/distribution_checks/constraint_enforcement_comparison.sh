#!/bin/bash

# Use MC_constr_graph_enumeration.py to count the number of chemical graphs satisfying constraints on which kinds of bonds
# are forbidden and which atoms can have covalent bonds with H.
# The first way is to explore chemical space without constraints and then counting the number of molecules that satisfy them.
# The second way is to explore chemical space with those constraints implicitly included in the sampling procedures.
# The results of the two approaches should agree.

for IMPLICIT_CONSTRAINT in FALSE TRUE
do
    echo "IMPLICIT CONSTRAINT: "$IMPLICIT_CONSTRAINT
    echo "Started: "$(date)
    python MC_constr_graph_enumeration.py $IMPLICIT_CONSTRAINT
    echo "Finished: "$(date)
done
