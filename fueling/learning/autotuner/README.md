# Learning Based Auto Tuning

## Setup Environment:

### start the fuel container and build py_proto inside the fuel container
```bash
cd apollo-fuel
./tools/login_container.sh
./tools/build_apollo.sh
```

## Run one iteration of the cost function at local

### Start cost computation server (**Inside the fuel container**)
```bash
bazel run //fueling/learning/autotuner/cost_computation:cost_computation_server
```

### Start simulation server (**Outside docker**)

1. mount bos with sudo
```bash
https://github.com/ApolloAuto/replay-engine/blob/master/scripts/auto_tuner/start_sim_service.sh#L5      line 5 and 6 here
```

2. Run js scripts
```bash
   cd ./replay-engine/scripts/auto_tuner
   sudo node auto_tuner_server.js
```

### Start bayesian optimization tuner (CostComputationClient side)

1. Setup parameter you want to tune in (Use MRAC as an example):
```bash
/apollo/modules/fule/fueling/autotuner/config/mrac_tuner_param_config.pb.txt
```

2. Run python scripts (For MRAC).
```bash
bazel run //fueling/learning/autotuner/tuner:bayesian_optimization_tuner
```

3. Or run other autotuner applications with
```bash
bazel run //fueling/learning/autotuner/tuner:bayesian_optimization_tuner -- --tuner_param_config_filename=<user defined tuner_param_config>
```


## Example of Calling Cost Computation Client
```python
from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient

result = CostComputationClient.compute_mrac_cost(
    # commit id
    "c693dd9e2e7910b041416021fcdb648cc4d8934d",
    # map of config_id to path_to_config pairs
    {
        "config_id1": {
           "/apollo/path/to/config1": "config1_pb2",
           "/apollo/path/to/config2": "config2_pb2"
         },
        "config_id2": {
           "/apollo/path/to/config2": "config2_pb2"
         },
    },
)
```
