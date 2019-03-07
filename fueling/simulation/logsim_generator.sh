#!/bin/bash

output_dir=$2
/apollo-simulator/bazel-bin/logsim_generator/logsim_generator_executable --input_dir=$1 --output_dir=$2 --scenario_map_dir=$3 --alsologtostderr

exit_code=$?
if [ $exit_code -ne 0 ]
then
  echo "Error: Exit with code $exit_code!"
fi
n=`ls $output_dir/*.json | wc -l`
echo "Finished generating $n scenarios."
