#!/usr/bin/env python
from absl import flags

flags.DEFINE_bool('is_backward', False, 'backward or forward for dynamic model training')
flags.DEFINE_bool('is_holistic', False, 'holistic or nonholistic for dynamic model training')

flags.DEFINE_bool('evaluation_vehicle_id', 'Mkz7', 'default vehicle_id for evaluation')
