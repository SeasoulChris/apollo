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

from job_processor import JobProcessor


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')


WORKERS = 5


class FuelJob(flask_restful.Resource):
    """Fuel job restful service"""

    def post(self):
        """Accept user request, verify and process."""
        try:
            request = flask.request.get_json()
            parser = json_format.Parse if isinstance(request, str) else json_format.ParseDict
            job_config = parser(request, JobConfig())
            http_code, msg = JobProcessor(job_config).process()
        except json_format.ParseError:
            http_code = HTTPStatus.BAD_REQUEST
            msg = 'job_config format error!'
        return json.dumps({'message': msg}), http_code


app = flask.Flask(__name__)
api = flask_restful.Api(app)
# TODO: We add a duplicate route for the restful service for a while, later the
# '/' route will be pointing to the portal '/index'.
api.add_resource(FuelJob, '/')
api.add_resource(FuelJob, '/proxy')

# @app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.tpl')


class ProductionApp(gunicorn.app.base.BaseApplication):
    """A wrapper to run flask app."""

    def __init__(self, flask_app):
        self.application = flask_app
        super(ProductionApp, self).__init__()

    def load_config(self):
        """Load config."""
        self.cfg.set('bind', '0.0.0.0:443')
        self.cfg.set('workers', WORKERS)
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
