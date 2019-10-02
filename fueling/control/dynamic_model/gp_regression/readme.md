
## Gaussian Process Regression
1. Setup Conda environment for label generation:
```bash
      $ conda env update --prune -f /apollo/modules/data/fuel/conda/py27.yaml
      $ source activate fuel-py27
```
2. Run label generation script:
```bash
      $ python /apollo/modules/data/fuel/fueling/control/dynamic_model/gp_regression/label_generation.py
```

3. Deactivate Conda environment
```bash
   conda deactivate
```

4. Run training script:
```bash
   ./fueling/control/dynamic_model/gp_regression/gp_main.sh
```
