#!/usr/bin/env python
"""Apollo Fuel web portal."""

from http import HTTPStatus
import json
import multiprocessing
import os

from absl import app as absl_app
from absl import flags
import flask
import google.protobuf.json_format as json_format
import gunicorn.app.base

from fueling.common.mongo_utils import Mongo
from apps.web_portal.job_processor import JobProcessor
from apps.web_portal.saas_job_arg_pb2 import SaasJobArg


flags.DEFINE_boolean('debug', False, 'Start local debug instance.')


################################# Web Handlers #################################
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/new_job/<string:job_type>')
def new_job(job_type):
    return flask.render_template('new_job.html', job_type=job_type)


@app.route('/submit_job', methods=['POST'])
def submit_job():
    try:
        request = flask.request.get_json()
        job_arg = json_format.ParseDict(request, SaasJobArg())
        http_code, msg = JobProcessor(job_arg).process()
    except BaseException:
        http_code = HTTPStatus.BAD_REQUEST
        msg = 'Wrong job argument'
    return msg, http_code


################################### Auto redirect to HTTPS in another process.

def http_to_https():
    http_app = flask.Flask(F'{__name__}_http')

    @http_app.route('/')
    @http_app.route('/<path:request_path>')
    def redirect_to_https(request_path=''):
        return flask.redirect(F'https://{flask.request.host}/{request_path}')

    http_app.run(host='0.0.0.0', port=8080)

################################### App Main ###################################
class ProductionApp(gunicorn.app.base.BaseApplication):
    """A wrapper to run flask app."""

    def __init__(self, flask_app):
        self.application = flask_app
        super(ProductionApp, self).__init__()

    def load_config(self):
        """Load config."""
        self.cfg.set('bind', '0.0.0.0:443')
        self.cfg.set('workers', 5)
        self.cfg.set('proc_name', 'BaeProxy')

        cwd = os.path.dirname(__file__)
        self.cfg.set('certfile', os.path.join(cwd, 'ssl_keys/cert.pem'))
        self.cfg.set('keyfile', os.path.join(cwd, 'ssl_keys/key.pem'))

    def load(self):
        """Load app."""
        return self.application


def main(argv):
    if flags.FLAGS.debug:
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        multiprocessing.Process(target=http_to_https, daemon=True).start()
        ProductionApp(app).run()


if __name__ == '__main__':
    absl_app.run(main)
