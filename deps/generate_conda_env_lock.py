#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Usage: conda list -n <ENV> | python3 generate_conda_env_lock.py input.yaml output.yaml

import collections
import glob
import os
import sys

input_yaml = sys.argv[1]
output_yaml = sys.argv[2]


def lib_name_to_import_name(lib):
    lib = lib.lower().replace('-', '_')
    if lib.startswith('python_'):
        lib = lib[7:]
    if lib.endswith('_py'):
        lib = lib[:-3]
    return lib


version_dict = {}
for line in sys.stdin.readlines():
    parts = line.split()
    if len(parts) >= 3:
        version_dict[lib_name_to_import_name(parts[0])] = parts[1]


usage_counter = collections.defaultdict(int)
fuel_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for path in ['fueling', 'apps']:
    for f in glob.glob(os.path.join(fuel_root, path, '**/*.py'), recursive=True):
        with open(f, encoding='utf-8') as fin:
            for line in fin:
                if line.startswith('import ') or line.startswith('from '):
                    usage_counter[line.split()[1].split('.')[0]] += 1
name_mappings = {
    'opencv': 'cv2',
    'keras_gpu': 'keras',
    'protobuf': 'google',
    'pyro_ppl': 'pyro',
    'pytorch': 'torch',
}
usage_counter.update({key : usage_counter[val] for key, val in name_mappings.items()})


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

            lib = line.split()[1]
            import_name = lib_name_to_import_name(lib.split(':')[2] if '::' in lib else lib)
            if import_name in version_dict:
                row = F'{heading} {lib} {equation} {version_dict[import_name]}'
                if import_name in usage_counter:
                    row += F'  # usage = {usage_counter[import_name]}'
                fout.write(row + '\n')
            else:
                fout.write(line)
