#!/usr/bin/env python

import flask
import flask_restful

from vehicle_calibration import VehicleCalibration

app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(VehicleCalibration, '/vehicle-calibration')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1080, debug=True,
            ssl_context=('deploy/cert.pem', 'deploy/key.pem'))
