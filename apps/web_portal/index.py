#!/usr/bin/env python
"""Apollo Fuel web portal."""

from http import HTTPStatus
import json
import os

from absl import app as absl_app
from absl import flags
import flask
import gunicorn.app.base
import requests


flags.DEFINE_boolean('debug', False, 'Start local debug instance.')
flags.DEFINE_string('kube_proxy', 'localhost', 'Kube proxy.')

# Web Handlers
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/new_job/<string:job_type>')
def new_job(job_type):
    return flask.render_template('new_job.html', job_type=job_type)


@app.route('/submit_job', methods=['POST'])
def submit_job():
    kube_proxy_url = F'http://localhost:8001'
    service_name = 'namespaces/default/services/http:spark-submitter-service:8000'
    handler = 'open-service'
    service_url = F'{kube_proxy_url}/api/v1/{service_name}/proxy/{handler}'
    try:
        resp = requests.post(service_url, json=json.dumps(flask.request.get_json()))
        http_code, msg = resp.status_code, resp.content
    except BaseException:
        http_code = HTTPStatus.BAD_REQUEST
        msg = 'Wrong job argument'
    return msg, http_code


# App Main
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
        ProductionApp(app).run()


if __name__ == '__main__':
    absl_app.run(main)
