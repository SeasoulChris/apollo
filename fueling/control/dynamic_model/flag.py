#!/usr/bin/env python
from absl import flags

# flags open to public via online service interface
flags.DEFINE_boolean('is_backward', False, 'backward or forward for dynamic model training')

# flags hold for internal use
flags.DEFINE_boolean('is_holistic', False, 'holistic or nonholistic for dynamic model training')
flags.DEFINE_string('evaluation_vehicle_id', 'Mkz7', 'default vehicle_id for evaluation')
