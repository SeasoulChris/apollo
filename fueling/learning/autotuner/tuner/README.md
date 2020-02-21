# Bayesian Optimization for Auto-tuning

## Environment Setup Steps:
```text
   conda config --add channels conda-forge
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
   python fueling/learning/autotuner/bayesian_optimization_tuner.py
```

Expected Results (Example):
```text
  result after: 5 steps are {'target': -6.118053703522349, 'params': {'x': 2.643480789490407, 'y': 1.3606425087500393}}
```
