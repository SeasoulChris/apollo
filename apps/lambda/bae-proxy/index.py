#!/usr/bin/env python
"""BAE proxy."""

import flask
import flask_restful

from absl import app as absl_app
from absl import flags
from vehicle_calibration import VehicleCalibration


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(VehicleCalibration, '/vehicle-calibration')


def main(argv):
    if flags.FLAGS.debug:
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        # Enable HTTPS for production.
        app.run(host='0.0.0.0', port=8080, ssl_context=('deploy/cert.pem', 'deploy/key.pem'))

if __name__ == '__main__':
    absl_app.run(main)
