#!/usr/bin/env python
"""BAE proxy."""

import flask
import flask_restful

from absl import app as absl_app
from absl import flags

from vehicle_calibration import VehicleCalibration


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')
flags.DEFINE_boolean('https', True, 'Enable HTTPS.')
flags.DEFINE_integer('port', 443, 'Port.')


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(VehicleCalibration, '/vehicle-calibration')


def main(argv):
    ssl_context = ('ssl_keys/cert.pem', 'ssl_keys/key.pem') if flags.FLAGS.https else None
    app.run(host='0.0.0.0', port=flags.FLAGS.port, debug=flags.FLAGS.debug, ssl_context=ssl_context)


if __name__ == '__main__':
    absl_app.run(main)
