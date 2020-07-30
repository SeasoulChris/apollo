 #!/usr/bin/env bash

 bazel run //fueling/planning/datasets:trajectory_perturbation_synthesizer_pipeline -- \
 --src_dir= \
 --output_dir= \
 --max_past_history_len=10 \
 --max_future_history_len=10 \
 --is_dumping_txt=False \
 --is_dumping_img=True
