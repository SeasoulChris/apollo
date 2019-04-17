"""Flag utils."""
#!/usr/bin/env python
import os
import sys

from absl import flags
import colored_glog as glog


class FlagUtils(object):
    """Utils to manage lifecycle of gflags."""
    INITED = False

    @classmethod
    def init_once(cls):
        """Init flags from flagfile."""
        if cls.INITED:
            return
        flagfile = os.environ.get('APOLLO_FLAGFILE')
        argv = list(sys.argv)
        if flagfile:
            flagfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', flagfile))
            if not os.path.exists(flagfile):
                glog.fatal('Cannot find flagfile {}'.format(flagfile))
                sys.exit(1)
            glog.info('Read flagfile {}'.format(flagfile))
            argv.append('--flagfile={}'.format(flagfile))
        flags.FLAGS(argv)
        cls.INITED = True


def get_flags():
    """Get flags."""
    FlagUtils.init_once()
    return flags.FLAGS
