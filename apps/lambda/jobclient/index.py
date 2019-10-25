
import json
import flask
from flask import request
import requests

from absl import flags
from absl import logging
from absl import app as absl_app

app = flask.Flask(__name__)
FLASK_DEBUG = 1

flags.DEFINE_string('fuel_proxy', 'https://apollofuel0.bceapp.com:8443',
                    'Endpoint of Apollo-Fuel proxy.')

flags.DEFINE_string(
    'certfile', '../bae-proxy/ssl_keys/cert.pem', 'certfication file')


@app.route('/')
def index():
    return flask.render_template('index.tpl')


@app.route('/submit_job', methods=['GET', 'POST'])
def submit_job():
    # request_json = request.get_json()
    # print(request.get_json())

    if request.json['bos']:
        storage = {'bos': {
            'bucket': request.json['bucket'],
            'access_key': request.json['access_key'],
            'access_secret': request.json['access_secret']}
        }
    elif request.json['blob']:
        storage = {'blob': {
            'blob_container': request.json['bucket'],
            'storage_account': request.json['access_key'],
            'storage_account_key': request.json['access_secret']}
        }

    request_dict = {
        'partner_id': request.json['partner_id'],
        'job_type': request.json['job_type'],
        'input_data_path': request.json['input_data_path'],
        'storage': storage
    }

    request_json = json.dumps(request_dict)

    print("parsed data is {}".format(request_json))

    response = requests.post('https://apollofuel0.bceapp.com:8443', json=request_json,
                             verify='../bae-proxy/ssl_keys/cert.pem')
    response = json.loads(response.json()) if response.json() else {}
    print('raw response: {}'.format(response))
    if response.ok:
        logging.info(response.get('message') or 'OK')
    else:
        logging.error(
            response.get('message') or
            'Request failed with HTTP code {}'.format(response.status_code))

    return "success"


def main(argv):
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    absl_app.run(main)
