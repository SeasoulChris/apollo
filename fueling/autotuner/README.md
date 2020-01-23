# Learning Based Auto Tuning

## Setup Environment:
```text
   conda config --add channels conda-forge
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
```
### Generate the gRPC interface from .proto service
-This is needed only if there're changes in the *.proto files:
```bash
    bash /apollo/apollo.sh build_py
```


## Run one iteration of the cost function at local:

### Start cost computation server
```bash
   cd /apollo/modules/data/fuel
   python fueling/autotuner/cost_computation/cost_computation_server.py
```

### Example of calling cost computation client
```python
from fueling.autotuner.client.cost_computation_client import CostComputationClient

weighted_score = CostComputationClient.compute_mrac_cost(
    # commit id
    "c693dd9e2e7910b041416021fcdb648cc4d8934d",
    [  # list of {path, config} pairs
        {"/apollo/path/to/config1": "c1_1", "/apollo/path/to/config2": "c1_2"},
        {"/apollo/path/to/config1": "c2_1"},
    ],
)
```
