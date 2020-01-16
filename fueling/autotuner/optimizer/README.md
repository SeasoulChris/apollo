# Bayesian Optimization for Auto-tuning

## Environment Setup Steps:
```text
   conda config --add channels conda-forge
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
   python fueling/autotuner/main.py 
```

Expected Results (Example):
```text
    optimizer 1 found a maximum value of: 0.9831053452160794
    optimizer 2 found a maximum value of: 0.9843973327748019
    optimizer 3 found a maximum value of: 0.9985724167335565
```