# Dynamic Model

## Feature Extraction
1. Upload data to BOS. Data includes:
   ```text
   - vehicle
        | - vehicle_param.pb.txt
        | - feature_key_conf.pb.txt
        | - folders_to_records
   ```
2. Run bash script at `apollo-fuel/`
   ```text
   ./fueling/control/dynamic_model/feature_extraction/feature_extraction.sh
   ```
3. Results are:
   * sample sets: `INTER_FOLDER/job_owner/job_id`, where `INTER_FOLDER` is defined in `./fueling/control/dynamic_model/feature_extraction/sample_set.py`
   * uniform distributed sets: `OUTPUT_FOLDER/job_owner/job_id/TODAY`, where`OUTPUT_FOLDER` is defined in `./fueling/control/dynamic_model/feature_extraction/uniform_set.py` and `TODAY` is the date when the script is run.
