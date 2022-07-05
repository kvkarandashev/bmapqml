#!/bin/bash

hooks_dir=$(dirname $0)/../.git/hooks

precomm_hook=$hooks_dir/pre-commit

cat > $precomm_hook << EOF
#!/bin/bash

# Check that black is installed.
if [[ "\$(pip show black 2>&1)" == WARNING:* ]]
then
    echo "Failed to find 'black' package used to format committed Python code."
    echo "Can be installed with 'pip install black'."
    exit 1
fi

FAIL=0

while read f
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
done <<< "\$(git status | awk '{if (\$1 == "modified:" || (\$1 == "new" && \$2 == "file:")) {print \$NF}}')"

if [ "\$FAIL" == "1" ]
then
    echo "Reformatting was needed, please re-attempt commit."
    exit 1
fi

EOF

chmod +x $precomm_hook
