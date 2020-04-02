# GP regression

This is the active development folder.

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
