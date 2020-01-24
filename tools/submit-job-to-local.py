#!/usr/bin/env python3

import os
import sys

try:
    from absl import app
    from absl import flags
    from absl import logging
except:
    print('Cannot import absl, you may need to run "sudo python3 -m pip install absl-py".')

flags.DEFINE_string('env', 'fuel-py36', 'Conda env name.', short_name='e')
flags.DEFINE_enum('log_verbosity', 'INFO', ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                  'Log verbosity.')
flags.DEFINE_string('main', None, 'Job entrypoint.')
flags.DEFINE_string('flags', None, 'Job flags.')
flags.DEFINE_integer('cpu', 1, 'Worker CPU cores.', short_name='c')


def main(argv):
    """Tool entrypoint."""
    if not os.path.exists('./fueling/'):
        logging.fatal('Must run from the apollo fule root directory.')
        sys.exit(1)
    os.system(F'./tools/submit-job-to-local.sh -e {flags.FLAGS.env} -c {flags.FLAGS.cpu} '
              F'-v {flags.FLAGS.log_verbosity} {flags.FLAGS.main} {flags.FLAGS.flags}')


if __name__ == '__main__':
    app.run(main)
