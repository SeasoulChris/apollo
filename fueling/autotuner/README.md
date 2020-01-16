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
   ./tools/submit-job-to-local.sh  fueling/autotuner/mrac_autotuner.py
```
