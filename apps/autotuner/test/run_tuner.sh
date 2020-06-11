source /home/libs/bash.rc
source /apollo/scripts/apollo_base.sh

bazel run //fueling/learning/autotuner/tuner:bayesian_optimization_tuner -- --cost_computation_service_url=costservice:50052
