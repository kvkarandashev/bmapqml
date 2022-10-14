#!/bin/bash

hooks_dir=$(dirname $0)/../.git/hooks

precomm_hook=$hooks_dir/pre-commit

cat > $precomm_hook << EOF
#!/bin/bash

function pip_package_check () {
    package=\$1
    language=\$2
    if [[ "\$(pip show \$package 2>&1)" == WARNING:* ]]
    then
        echo "Failed to find '\$package' package used to format committed \$language code."
        echo "Taking the liberty to attempt installing \$package'."
        pip install \$package
        echo "Please re-attempt commit."
        exit 1
    fi
}


FAIL=0

while read f
do
    if [[ \$f == *.py ]]
    then
        pip_package_check black Python
        if ! \$(python \$(which black) \$f --check -q)
        then
            echo "Reformatting \$f."
            black \$f
            FAIL=1
        fi
    fi
    if [[ \$f == *.f90 ]]
    then
        pip_package_check fprettify Fortran90
        if [ "\$(fprettify --diff \$f | wc -l)" != "0" ]
        then
            echo "Reformatting \$f."
            fprettify \$f
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
