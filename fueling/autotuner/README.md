# Learning Based Auto Tuning

## Setup Environment:
```text
   conda config --add channels conda-forge
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
```

## Test auto tuner at local:
```bash
    cd /apollo/modules/data/fuel
   ./tools/submit-job-to-local.sh  fueling/autotuner/mrac_autotuner.py --commit=<apollo commit id>
```

## Generate the Python code for sim_service
If there's change in the sim_service.proto, generating new python code is needed:
```bash
   cd /apollo/modules/data/fuel/fueling/autotuner
   python -m grpc_tools.protoc -I./proto --python_out=./client --grpc_python_out=./client ./proto/sim_service.proto
```

The generated code files are called sim_service_pb2.py and sim_service_pb2_grpc.py.
Once generated, change the import path in sim_service_pb2_grpc.py to absolute path
```text
import fueling.autotuner.client.sim_service_pb2 as sim__service__pb2
```
