#!/usr/bin/env bash

set -e

# Preapre: Goto fuel root, checkout latest code.
#cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

#git remote update
#git reset --hard origin/master

# Job: map generate base_map.txt
JOB="fueling/map/generate_base_map.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 10g ${JOB}

# Job: map generate for sim_map.txt or routing_map.txt
JOB="fueling/map/generate_sim_routing_map.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 10g ${JOB}