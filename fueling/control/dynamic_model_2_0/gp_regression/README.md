# GP regression

This is the active development folder.

We need gpytorch latest master (instead of Conda Forge Installed 1.0.1) to properly train and save corresponding models, which requires `pytorch>=1.5`, while the current prediction/planning modules runs on `pytorch v1.2`, until this version conflict been fully resolved, please run following command to everytime when starts `fuel` docker.

Install `gpytorch` from master
```
pip install -e git+git://github.com/cornellius-gp/gpytorch.git@master#egg=gpytorch
```

Use
```
 bazel run fueling/control/dynamic_model_2_0/gp_regression:train
```
 To train the model, model output default saved in
 ```
 fueling/control/dynamic_model_2_0/gp_regression/testdata/gp_model_output
 ```


 Use
 ```
  bazel run fueling/control/dynamic_model_2_0/gp_regression:evaluation
 ```
  To evaluate the model performance, evaluation output default saved in
  ```
  fueling/control/dynamic_model_2_0/gp_regression/testdata/evaluation_result
  ```
