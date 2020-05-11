#!/usr/bin/env python
"""Serve data imported in MongoDB."""

from datetime import timezone
import collections
import datetime
import os

from absl import app as absl_app
from absl import flags
import flask
import flask_socketio
import gunicorn.app.base
import pymongo

from fueling.common.kubectl_utils import Kubectl
from fueling.common.mongo_utils import Mongo
from fueling.data.proto.record_meta_pb2 import RecordMeta
# uncomment below for testing
# import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
import fueling.common.redis_utils as redis_utils

import apps.k8s.warehouse.display_util as display_util
import apps.k8s.warehouse.metrics_util as metrics_util
import apps.k8s.warehouse.records_util as records_util


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')


HOST = '0.0.0.0'
PORT = 8000
WORKERS = 5
PAGE_SIZE = 30
METRICS_PV_PREFIX = 'apps.warehouse.pv.'
TIMEZONE = 'America/Los_Angeles'

app = flask.Flask(__name__)
app.secret_key = str(datetime.datetime.now())
app.jinja_env.filters.update(display_util.utils)
socketio = flask_socketio.SocketIO(app)

# apply the kubenetes config file
# uncomment below for testing
# kubectl = Kubectl(file_utils.fuel_path('apps/k8s/warehouse/kubectl.conf'))
# comment below for testing
kubectl = Kubectl()


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
        return flask.render_template('base.html')

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
        task_records[doc['dir']].append(
            proto_utils.dict_to_pb(doc, RecordMeta()))
    tasks = [records_util.CombineRecords(records)
             for records in task_records.values()]
    tasks.sort(key=lambda task: task.dir, reverse=True)
    return flask.render_template(
        'records.html', page_count=page_count, prefix=prefix, current_page=page_idx,
        records=tasks,
        is_tasks=True)


@app.route('/task/<path:task_path>')
def task_hdl(task_path):
    """Handler of the task detail page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'task')
    docs = Mongo().record_collection().find(
        {'dir': os.path.join('/', task_path)})
    records = [proto_utils.dict_to_pb(doc, RecordMeta()) for doc in docs]
    task = records_util.CombineRecords(records)
    return flask.render_template(
        'record.html', record=task, sub_records=records)


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
        'records.html', page_count=page_count, current_page=page_idx, records=records)


@app.route('/record/<path:record_path>')
def record_hdl(record_path):
    """Handler of the record detail page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'record')
    doc = Mongo().record_collection().find_one(
        {'path': os.path.join('/', record_path)})
    record = proto_utils.dict_to_pb(doc, RecordMeta())
    return flask.render_template('record.html', record=record)


# TODO(Andrew):
# 1. Filter the items, only show fuel jobs like job-<job_id>-<job_name>-<timestamp>-driver.
#    We don't want to expose long-run deployments like warehouse and simulation services.
#    It's also risky if we allow users to kill pods in the future.
# 2. Reverse the job list, because people always care more about recent jobs.
#    Show more information in job title bar, such as the phase.
#    so that people know the job status without expanding the job panel.
#    If it makes the title bar too long, just remove the -<timestamp>-driver part from job name :)
# 3. As you already know, besides Logs button, we can also add Info button,
#    which shows result of kubectl describe pod <name>. And Stop button,
#    which triggers kubectl delete pod <name>.
@app.route('/jobs')
def jobs_hdl():
    """Handler of the pod list page"""
    res = kubectl.get_pods()
    jobs_dict = {}
    curr_datetime = datetime.datetime.now(timezone.utc)
    for pod in res:
        namespace = pod.metadata.namespace
        podname = pod.metadata.name
        if not podname.startswith('job-'):
            continue
        phase = pod.status.phase
        creation_timestamp = pod.metadata.creation_timestamp.replace(
            tzinfo=timezone.utc)
        duration_ns = (curr_datetime - creation_timestamp).seconds * 1e9
        # executors
        if pod.metadata.owner_references is not None:
            appuid = pod.metadata.owner_references[0].uid
            if appuid not in jobs_dict:
                jobs_dict[appuid] = {}
                jobs_dict[appuid]['pods'] = []
            # executors' info
            jobs_dict[appuid]['pods'].append(({
                'podname': podname,
                'phase': phase,
                'creation_timestamp': creation_timestamp.timestamp(),
                'duration_ns': duration_ns
            }))
        # drivers
        else:
            poduid = pod.metadata.uid
            if poduid not in jobs_dict:
                jobs_dict[poduid] = {}
                jobs_dict[poduid]['pods'] = []
            # extract job_owner for future filtering
            for env in pod.spec.containers[0].env:
                if env.name == 'PYSPARK_APP_ARGS':
                    for v in env.value.split(' '):
                        if '=' in v:
                            arg_type, arg_value = v.split('=')
                            if arg_type == '--job_owner':
                                jobs_dict[poduid]['owner'] = arg_value
                            elif arg_type == '--job_id':
                                jobs_dict[poduid]['job_id'] = arg_value
            # drivers' info
            jobs_dict[poduid]['namespace'] = namespace
            jobs_dict[poduid]['name'] = podname
            jobs_dict[poduid]['creation_timestamp'] = creation_timestamp.timestamp()
            jobs_dict[poduid]['phase'] = phase
            jobs_dict[poduid]['pods'].append({
                'podname': podname,
                'phase': phase,
                'creation_timestamp': creation_timestamp.timestamp(),
                'duration_ns': duration_ns
            })
    sorted_job_list = sorted(list(jobs_dict.items()),
                             key=lambda x: x[1]['creation_timestamp'],
                             reverse=True)
    return flask.render_template('jobs.html', jobs_list=sorted_job_list)


# TODO(Andrew):
# 1. For the log page, it's OK to load and show all logs at once as a start. But in the
# future it would be every fancy if it updates at realtime! Just like "kubectl logs -f <pod>".
# 2. We may need to add the filter to the logs. As Spark itself outputs a lot of boring and noisy
# logs, while we just want to see logs from our Python code.
@app.route('/pod_log/<path:pod_name>/<path:namespace>')
def pod_log_hdl(pod_name, namespace='default'):
    """Handler of the pod log page"""
    logs = kubectl.logs(pod_name=pod_name, namespace=namespace)
    return flask.render_template('pod_log.html', logs=logs)


@app.route('/bos-ask', methods=['POST'])
def bos_ask():
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'bos-ask')
    if flask.request.form.get('pin') != 'woyouyitouxiaomaolv':
        return ''
    return '{}{}'.format(
        os.environ.get('BOS_ASK_ACCESS'),
        os.environ.get('BOS_ASK_SECRET'))


@app.route('/metrics', methods=['GET'])
def metrics_hdl():
    """Handler of the redis metrics."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'metrics')
    prefix = flask.request.args.get('prefix') or ''
    metrics = metrics_util.get_metrics_by_prefix(prefix)
    return flask.render_template(
        'metrics.html', prefix=prefix, metrics=metrics)


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
    return flask.render_template('plot.html', data={'key': redis_key, 'type': plot_type})


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


def main(argv):
    socketio.run(app, HOST, PORT, debug=flags.FLAGS.debug)


if __name__ == '__main__':
    absl_app.run(main)

