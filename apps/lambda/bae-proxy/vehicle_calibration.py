#!/usr/bin/env python

import httplib

from flask_restful import Resource, reqparse

parser = reqparse.RequestParser()
parser.add_argument('proto')


class VehicleCalibration(Resource):
    """Vehicle Calibration restful service"""

    def get(self):
        return "OK"

    def post(self):
        proto_text = parser.parse_args().get('proto')
        # TODO(xiaoxq): This is just an echo for now.
        # 1. Parse the text to proto.
        # 2. Sanity check.
        # 3. Send notification and operation.
        return proto_text, httplib.ACCEPTED
