# Learning Based Auto Tuning

## Setup Environment:
### Generate the gRPC interface from .proto service
This is needed only if there're changes in the \*.proto files:
```bash
    bash /apollo/apollo.sh build_py
```

```text
   conda config --add channels conda-forge
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
```



## Run one iteration of the cost function at local:

### Start cost computation server
```bash
   cd /apollo/modules/data/fuel
   chmod a+x fueling/autotuner/cost_computation/cost_computation_server.py
   ./fueling/autotuner/cost_computation/cost_computation_server.py
```

### Start bayesian optimization tuner (CostComputationClient side)

1. Setup parameter you want to tune in (Use MRAC as an example):
```bash
/apollo/modules/fule/fueling/autotuner/config/mrac_tuner_param_config.pb.txt
```

2. Run python scripts (For MRAC).
```bash
   cd /apollo/modules/data/fuel
   chmod a+x fueling/autotuner/tuner/bayesian_optimization_tuner.py
   ./fueling/autotuner/tuner/bayesian_optimization_tuner.py
```

3. Or run other autotuner applications with
```bash
   cd /apollo/modules/data/fuel
   python fueling/autotuner/tuner/bayesian_optimization_tuner.py --tuner_param_config_filename=<user defined tuner_param_config>
```

### Start replay-engine grpc server:

1. mount bos with sudo
```bash
https://github.com/ApolloAuto/replay-engine/blob/master/scripts/auto_tuner/start_sim_service.sh#L5      line 5 and 6 here
```

2. Run js scripts (**Outside docker**)
```bash
   cd ./replay-engine/scripts/auto_tuner
   sudo node auto_tuner_server.js
```
## Example of Calling Cost Computation Client
```python
from fueling.autotuner.client.cost_computation_client import CostComputationClient

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
