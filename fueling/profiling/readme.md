# Profiling

## 1. Control Profiling Application on Data Pipeline (for Road-Test)

(under construction)

## 2. Control Profiling Application on Dreamland (for Simulation-Test)

### Command Line Format
(The following codes are executed under the path **/apollo/modules/data/fuel**)

```bash
# Setup the record/bag input directory and profiling results output directory.
# For example,
data_path='/apollo/modules/data/fuel/testdata/profiling/control_profiling/Sim_Test'
results_path='/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated'

# Run the control profiling metrics command (generating performance_grading.json file)
./tools/submit-job-to-local.sh fueling/profiling/control_profiling_metrics.py -ctl_metrics_input_path_local $data_path -ctl_metrics_output_path_local $results_path

# Run the control profiling visualization command (generating feature_data.json file)
./tools/submit-job-to-local.sh fueling/profiling/control_profiling_visualization.py -ctl_visual_input_path_local $results_path -ctl_visual_output_path_local $results_path --ctl_visual_simulation_only_test
```

### Output Data Format

#### 1. Control Performance Grading .json file

The dictionary data structure included in the **...control_performance_grading.json** file contains 40 grading metrics (each metrics contains several fields) with format of

`{"grading_metrics1": {"grading_field1": value, "grading_field2": value, ...} ...}`

For example,

`{"station_err_std": {"score": 0.075403, "sample_size": 3483}, "station_err_peak": {"score": 0.717374, "sample_size": 3499, "timestamp": 4.75}, ...}`

#### 2. Control Feature Data .json file

The dictionary data structure included in the **...control_data_feature.json** file contains 46 control features (each feature contains a list of raw data value) and 1 additional "labels" dict structure with format of

`{"feature_key1": [value1, value2, value3, ...], "feature_key2": [value1, value2, value3, ...], ... , "labels": {"x_label": feature_key1, "y_label": [feature_key2, feature_key3, feature_key4, ...]}}`

For example,

`{"station_reference": [-0.202164, -0.134776, -0.134776, -0.067388, -0.067388, ...], ... ,
"labels": {"x_label": "timestamp_sec", "y_label": ["station_reference", "speed_reference", ...]}}`

#### 3. Control Feature Statistics .json file

The dictionary data structure included in the **...control_data_statistics.json** file contains 46 control features (each feature contains several statistic fields) with format of

`{"feature_key1": {"statistic_field1": value1, "statistic_field2": value2, ...}, ... }`

For example,

`{"station_reference": {"mean": -0.138802, "standard deviation": 0.400271}, ... }`
