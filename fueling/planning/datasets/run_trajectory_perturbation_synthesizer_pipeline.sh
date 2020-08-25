 #!/usr/bin/env bash

 bazel run //fueling/planning/datasets:trajectory_perturbation_synthesizer_pipeline -- \
 --src_dir=/fuel/local_training_data/validation \
 --output_dir=/fuel/local_training_data/synthesized \
 --max_past_history_len=10 \
 --max_future_history_len=10 \
 --is_dumping_txt=False \
 --is_dumping_img=True
