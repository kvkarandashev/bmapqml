#!/bin/bash

hooks_dir=$(dirname $0)/../.git/hooks

cat > $hooks_dir/pre-commit << EOF
#!/bin/bash

# Check that black is installed.
if [[ "\$(pip show black)" == WARNING:* ]]
then
    echo "Failed to find 'black' package used to format committed Python code."
    echo "Can be installed with 'pip install black'."
    exit 1
fi

FAIL=0

git status | awk '{if (\$1 == "modified:" || (\$1 == "new" && \$2 == "file:")) {print \$NF}}' | while read f
do
    if [[ \$f == *.py ]]
    then
        if ! \$(black \$f --check -q)
        then
            echo "Reformatting \$f."
            black \$f
            FAIL=1
        fi
    fi
done

if [ "\$FAIL" == "1" ]
then
    echo "Reformatting was needed, please re-attempt commit."
    exit 1
fi

EOF
