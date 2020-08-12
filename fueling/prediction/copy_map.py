#!/usr/bin/env python
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.context_utils as context_utils


class CopyMap(BasePipeline):
    """Delete dirs distributed on several workers pipeline."""

    def run(self):
        """Run prod."""
        map_path = os.path.join(self.FLAGS.get('input_path'), 'map')
        map_path = self.our_storage().abs_path(map_path)
        map_list = os.listdir(map_path)
        assert len(map_list) == 1
        map_region_path = os.path.join(map_path, map_list[0])

        map_target_path = self.our_storage().abs_path('code/baidu/adu-lab/apollo-map/')
        if context_utils.is_local():
            map_target_path = '/apollo/modules/map/data/'
        copy_map_command = 'cp -r {} {}'.format(map_region_path, map_target_path)
        os.system(copy_map_command)


if __name__ == '__main__':
    CopyMap().main()
