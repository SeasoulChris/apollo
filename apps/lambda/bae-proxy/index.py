#!/usr/bin/env python
"""BAE proxy."""

from http import HTTPStatus
import json

from absl import app as absl_app
from absl import flags
import flask
import flask_restful
import google.protobuf.json_format as json_format
import gunicorn.app.base

from modules.tools.fuel_proxy.proto.job_config_pb2 import JobConfig

from fueling.common.partners import partners
from vehicle_calibration import VehicleCalibration


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')
flags.DEFINE_integer('workers', 5, 'Workers to run.')


class FuelJob(flask_restful.Resource):
    """Fuel job restful service"""

    def post(self):
        """Accept user request, verify and process."""
        # 1. Parse request.
        try:
            request = flask.request.get_json()
            parser = json_format.Parse if isinstance(request, str) else json_format.ParseDict
            job_config = parser(request, JobConfig())
        except json_format.ParseError:
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        # 2. User authentication.
        if job_config.partner_id not in partners:
            msg = 'Sorry, you are not authorized to access this service!'
            return json.dumps({'message': msg}), HTTPStatus.UNAUTHORIZED
        # 3. Dispatch jobs.
        if job_config.job_type == JobConfig.VEHICLE_CALIBRATION:
            return VehicleCalibration.process(job_config)
        return json.dumps({'message': 'Unsupported job type'}), HTTPStatus.BAD_REQUEST


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(FuelJob, '/')


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
