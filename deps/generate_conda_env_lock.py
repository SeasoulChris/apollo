#!/usr/bin/env python3
# Usage: conda list -n <ENV> | python3 generate_conda_env_lock.py input.yaml output.yaml

import sys

input_yaml = sys.argv[1]
output_yaml = sys.argv[2]

version_dict = {}
for line in sys.stdin.readlines():
    parts = line.split()
    if len(parts) >= 3:
        version_dict[parts[0].lower()] = parts[1]

with open(input_yaml, 'r') as fin:
    with open(output_yaml, 'w') as fout:
        for line in fin:
            heading = ''
            equation = ''
            if line.startswith('  -'):
                # Conda lib
                heading = '  -'
                equation = '='
            elif line.startswith('    -'):
                # Pip lib
                heading = '    -'
                equation = '=='
            else:
                fout.write(line)
                continue

            lib_name = line.split()[1]
            channeless_lib_name = lib_name.split(':')[2] if '::' in lib_name else lib_name
            channeless_lib_name = channeless_lib_name.lower()
            if channeless_lib_name in version_dict:
                fout.write(F'{heading} {lib_name} {equation} {version_dict[channeless_lib_name]}\n')
            else:
                fout.write(line)
