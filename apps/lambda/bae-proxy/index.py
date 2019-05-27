#!/usr/bin/env python
"""BAE proxy."""

import flask
import flask_restful

from absl import app as absl_app
from absl import flags
import gunicorn.app.base

from vehicle_calibration import VehicleCalibration


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')
flags.DEFINE_integer('workers', 5, 'Workers to run.')


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(VehicleCalibration, '/vehicle-calibration')


class ProductionApp(gunicorn.app.base.BaseApplication):
    """A wrapper to run flask app."""
    def __init__(self, flask_app):
        self.application = flask_app
        super(ProductionApp, self).__init__()

    def load_config(self):
        """Load config."""
        self.cfg.set('bind', '0.0.0.0:443')
        self.cfg.set('workers', flags.FLAGS.workers)
        self.cfg.set('proc_name', 'BaeProxy')
        self.cfg.set('certfile', 'ssl_keys/cert.pem')
        self.cfg.set('keyfile', 'ssl_keys/key.pem')

    def load(self):
        """Load app."""
        return self.application


def main(argv):
    if flags.FLAGS.debug:
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        ProductionApp(app).run()


if __name__ == '__main__':
    absl_app.run(main)
