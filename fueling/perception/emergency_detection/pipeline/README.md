# Record Image Parser Pipeline

This is a pipeline that can parse images from a record folder

## Usage

1. In apollo-fuel docker, run ```bash bazel run //fueling/streaming:serialize_records -- --cloud ```.
   This will start a job called serialize_records which acts like a server.

2. In apollo-fuel docker, run 
   ```bash
   bazel run //fueling/perception/emergency_detection/pipeline:record_image_parser -- --cloud --record_folder=<record_file_path>
   ```
   eg: 
   ```bash
   bazel run //fueling/perception/emergency_detection/pipeline:record_image_parser -- --cloud --record_folder=public-test/2020/2020-08-24/2020-08-24-14-58-13
   ```

3. Serialize_records will first process the raw record files to serialized messages and store them at: /mnt/bos/modules/streaming/data

4. DecodeVideoPipeline will then start to extract images from those messages and store them at: mnt/bos/modules/perception/emergency_detection/processed/images_from_video