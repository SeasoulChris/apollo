# Learning Based Auto Tuning

## Setup Environment:
### Generate the gRPC interface from .proto service
-This is needed only if there're changes in the \*.proto files:
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
   python fueling/autotuner/cost_computation/cost_computation_server.py
```

### Start bayesian optimization tuner (CostComputationClient side)
```bash
   cd /apollo/modules/data/fuel
   python fueling/autotuner/cost_computation/cost_computation_server.py
```
