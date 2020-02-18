## Gaussian Process Regression

1. Setup Conda environment for label generation:

   ```bash
   conda env update --prune -f /fuel/conda/py27-cyber.yaml
   source activate fuel-py27-cyber
   ```

1. Run label generation script:

   ```bash
   python /fuel/fueling/control/dynamic_model/gp_regression/label_generation.py
   ```

1. Deactivate Conda environment

   ```bash
   conda deactivate
   ```

1. Run training script:

   ```bash
   conda env update --prune -f /fuel/conda/py36-pyro.yaml
   ./fueling/control/dynamic_model/gp_regression/gp_main.sh
   ```
