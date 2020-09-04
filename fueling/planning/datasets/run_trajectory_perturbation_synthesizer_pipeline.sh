 #!/usr/bin/env bash

 bazel run //fueling/planning/datasets:trajectory_perturbation_synthesizer_pipeline -- \
 --src_dir= \
 --output_dir= \
 --max_past_history_len=10 \
 --max_future_history_len=10 \
 --is_lateral_or_longitudinal_synthesizing=True \
 --perturbate_normal_direction_range=-2 \
 --perturbate_normal_direction_range=2 \
 --perturbate_tangential_direction_range=0.2 \
 --perturbate_tangential_direction_range=2.0 \
 --ref_cost=1.0 \
 --elastic_band_smoothing_cost=50.0 \
 --max_curvature=0.3 \
 --is_dumping_txt=False \
 --is_dumping_img=False
