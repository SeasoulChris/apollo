# Dynamic Model

## Feature Extraction
1. Upload data to BOS. Data includes:
   ```text
   - vehicle
        | - vehicle_param.pb.txt
        | - feature_key_conf.pb.txt
        | - folders_to_records
   ```
   Some sample data set has been uploaded at `mnt/bos/modules/control/learning_based_model/test_data`
2. Bazel run at apollo fuel docker `fuel_XXX:/fuel`, for example
   ```text
   bazel run //fueling/control/dynamic_model:dynamic_model -- --cloud --input_data_path=modules/control/learning_based_model/test_data
   ```
3. Results are:
   * sample sets: `mnt/bos/sample_output_folder/job_owner/direction/job_id`,
   where `sample_output_folder` is defined in `fuel/fueling/control/dynamic_model/conf/model_config.py`
   * uniform distributed sets: `mnt/bos/uniform_output_folder/job_owner/direction/job_id`,
   where`uniform_output_folder` is defined in `fuel/fueling/control/dynamic_model/conf/model_config.py`.
