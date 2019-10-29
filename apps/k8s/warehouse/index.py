#!/usr/bin/env python
"""Serve data imported in MongoDB."""

import collections
import datetime
import os

from absl import app as absl_app
from absl import flags
import flask
import flask_restful
import flask_socketio
import gunicorn.app.base
import pymongo

from fueling.common.mongo_utils import Mongo
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta
import fueling.common.proto_utils as proto_utils
import fueling.common.redis_utils as redis_utils

from res_map_lookup import MapLookup
import display_util
import records_util
import metrics_util


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')


HOST = '0.0.0.0'
PORT = 8000
WORKERS = 5
PAGE_SIZE = 20
METRICS_PV_PREFIX = 'apps.warehouse.pv.'

app = flask.Flask(__name__)
app.secret_key = str(datetime.datetime.now())
app.jinja_env.filters.update(display_util.utils)
socketio = flask_socketio.SocketIO(app)


@app.route('/')
@app.route('/tasks/<prefix>/<int:page_idx>')
def tasks_hdl(prefix='small-records', page_idx=1):
    """Handler of the task list page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'tasks')
    mongo_col = Mongo().record_collection()
    query = {'dir': {'$regex': '^/mnt/bos/' + prefix}}
    task_dirs = {doc['dir'] for doc in mongo_col.find(query, {'dir': 1})}
    page_count = (len(task_dirs) + PAGE_SIZE - 1) // PAGE_SIZE
    if page_idx > page_count:
        flask.flash('Page index out of bound')
        return flask.render_template('base.tpl')

    offset = PAGE_SIZE * (page_idx - 1)
    task_dirs = sorted(list(task_dirs), reverse=True)
    query = {'dir': {'$in': task_dirs[offset: offset + PAGE_SIZE]}}
    kFields = {
        'dir': 1,
        'header.begin_time': 1,
        'header.end_time': 1,
        'header.size': 1,
        'hmi_status.current_mode': 1,
        'hmi_status.current_map': 1,
        'hmi_status.current_vehicle': 1,
        'disengagements': 1,
        'drive_events': 1,
        'stat.mileages': 1,
    }
    task_records = collections.defaultdict(list)
    for doc in mongo_col.find(query, kFields):
        task_records[doc['dir']].append(proto_utils.dict_to_pb(doc, RecordMeta()))
    tasks = [records_util.CombineRecords(records) for records in task_records.values()]
    tasks.sort(key=lambda task: task.dir, reverse=True)
    return flask.render_template(
        'records.tpl', page_count=page_count, prefix=prefix, current_page=page_idx, records=tasks,
        is_tasks=True)


@app.route('/task/<path:task_path>')
def task_hdl(task_path):
    """Handler of the task detail page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'task')
    docs = Mongo().record_collection().find({'dir': os.path.join('/', task_path)})
    records = [proto_utils.dict_to_pb(doc, RecordMeta()) for doc in docs]
    task = records_util.CombineRecords(records)
    return flask.render_template('record.tpl', record=task, sub_records=records)


@app.route('/records')
@app.route('/records/<int:page_idx>')
def records_hdl(page_idx=1):
    """Handler of the record list page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'records')
    kFields = {
        'path': 1,
        'header.begin_time': 1,
        'header.end_time': 1,
        'header.size': 1,
        'hmi_status.current_mode': 1,
        'hmi_status.current_map': 1,
        'hmi_status.current_vehicle': 1,
        'disengagements': 1,
        'drive_events': 1,
        'stat.mileages': 1,
    }
    kSort = [('header.begin_time', pymongo.DESCENDING)]

    docs = Mongo().record_collection().find({}, kFields)
    page_count = (docs.count() + PAGE_SIZE - 1) // PAGE_SIZE
    offset = PAGE_SIZE * (page_idx - 1)
    records = [proto_utils.dict_to_pb(doc, RecordMeta())
               for doc in docs.sort(kSort).skip(offset).limit(PAGE_SIZE)]
    return flask.render_template(
        'records.tpl', page_count=page_count, current_page=page_idx, records=records)


@app.route('/record/<path:record_path>')
def record_hdl(record_path):
    """Handler of the record detail page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'record')
    doc = Mongo().record_collection().find_one({'path': os.path.join('/', record_path)})
    record = proto_utils.dict_to_pb(doc, RecordMeta())
    return flask.render_template('record.tpl', record=record)


@app.route('/bos-ask', methods=['POST'])
def bos_ask():
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'bos-ask')
    if flask.request.form.get('pin') != 'woyouyitouxiaomaolv':
        return ''
    return '{}{}'.format(os.environ.get('BOS_ASK_ACCESS'), os.environ.get('BOS_ASK_SECRET'))


@app.route('/metrics', methods=['GET'])
def metrics_hdl():
    """Handler of the redis metrics."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'metrics')
    prefix = flask.request.args.get('prefix') or ''
    metrics = metrics_util.get_metrics_by_prefix(prefix)
    return flask.render_template('metrics.tpl', prefix=prefix, metrics=metrics)


@app.route('/metrics_ajax')
def metrics_ajax():
    """Handler of ajax request from client"""
    # TODO(Longtao): remove after the socketio connection issue is fixed
    return metrics_util.get_metrics_by_prefix(flask.request.args.get('prefix'))


@socketio.on('client_request_metrics_event')
def metrics_request_event(message):
    """Handler of socketio client request"""
    server_response_channel = 'server_response_metrics'
    metrics = metrics_util.get_metrics_by_prefix(message['prefix'])
    flask_socketio.emit(server_response_channel, metrics)


@app.route('/plot_img', methods=['GET'])
def plot_img():
    """Handler of profiling plot request"""
    redis_key = flask.request.args.get('key')
    plot_type = flask.request.args.get('type')
    return flask.render_template('plot.tpl', data={'key': redis_key, 'type': plot_type})


class FlaskApp(gunicorn.app.base.BaseApplication):
    """A wrapper to run flask app."""

    def __init__(self, flask_app):
        flask_app.debug = flags.FLAGS.debug
        self.application = flask_app
        super(FlaskApp, self).__init__()

    def load_config(self):
        """Load config."""
        self.cfg.set('bind', '{}:{}'.format(HOST, PORT))
        self.cfg.set('workers', WORKERS)
        self.cfg.set('proc_name', 'ApolloData')

    def load(self):
        """Load app."""
        return self.application


api = flask_restful.Api(app)
# As there might be negative values which are not supported by float type, we
# accept them as string and convert in code.
api.add_resource(MapLookup, '/map-lookup/<string:lat>/<string:lon>')


def main(argv):
    socketio.run(app, HOST, PORT, debug=flags.FLAGS.debug)


if __name__ == '__main__':
    absl_app.run(main)
