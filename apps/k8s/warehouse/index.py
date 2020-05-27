#!/usr/bin/env python
"""Serve data imported in MongoDB."""

from datetime import timezone
import collections
import datetime
import dateutil.parser
import json
import os

from absl import app as absl_app
from absl import flags
from ansi2html import Ansi2HTMLConverter
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
        task_records[doc['dir']].append(proto_utils.dict_to_pb(doc, RecordMeta()))
    tasks = [records_util.CombineRecords(records) for records in task_records.values()]
    tasks.sort(key=lambda task: task.dir, reverse=True)
    return flask.render_template(
        'records.html', page_count=page_count, prefix=prefix, current_page=page_idx, records=tasks,
        is_tasks=True)


@app.route('/task/<path:task_path>')
def task_hdl(task_path):
    """Handler of the task detail page."""
    redis_utils.redis_incr(METRICS_PV_PREFIX + 'task')
    docs = Mongo().record_collection().find({'dir': os.path.join('/', task_path)})
    records = [proto_utils.dict_to_pb(doc, RecordMeta()) for doc in docs]
    task = records_util.CombineRecords(records)
    return flask.render_template('record.html', record=task, sub_records=records)


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
    doc = Mongo().record_collection().find_one({'path': os.path.join('/', record_path)})
    record = proto_utils.dict_to_pb(doc, RecordMeta())
    return flask.render_template('record.html', record=record)


@app.route('/jobs')
def jobs_hdl():
    """Handler of the pod list page"""

    def extract_pod_info(pod):
        """extract pod data from json"""
        namespace = pod['metadata']['namespace']
        podname = pod['metadata']['name']
        phase = pod['status']['phase']
        creation_timestamp = (dateutil.parser.parse(pod['metadata']['creationTimestamp'])
                              .replace(tzinfo=timezone.utc))
        curr_datetime = datetime.datetime.now(timezone.utc)
        duration_ns = (curr_datetime - creation_timestamp).seconds * 1e9
        pod_uid = pod['metadata']['uid']
        parent_uid = ''
        spark_job_owner = ''
        spark_job_id = ''
        # only executor has an parent_id indicating the uid of corresponding driver
        if 'ownerReferences' in pod['metadata']:
            parent_uid = pod['metadata']['ownerReferences'][0]['uid']
        else:
            # extract job_owner for future filtering
            try:
                for env in pod['spec']['containers'][0]['env']:
                    if env['name'] == 'PYSPARK_APP_ARGS':
                        for v in env['value'].split(' '):
                            if '=' in v:
                                arg_type, arg_value = v.split('=')
                                if arg_type == '--job_owner':
                                    spark_job_owner = arg_value
                                elif arg_type == '--job_id':
                                    spark_job_id = arg_value
            except Exception as ex:
                pass
        return {'namespace': namespace,
                'podname': podname,
                'phase': phase,
                'creation_timestamp': creation_timestamp,
                'duration_ns': duration_ns,
                'pod_uid': pod_uid,
                'parent_uid': parent_uid,
                'spark_job_id': spark_job_id,
                'spark_job_owner': spark_job_owner}

    jobs_dict = {}
    kubectl_jobs = set()

    def save_pod_info(podinfo, podfrom):
        """save pod info to dict
        podfrom: ['kubectl', 'mongodb']
        Notice: load the kubectl jobs frist, then mongodb jobs
        """
        # only display fuel jobs
        if not podinfo['podname'].startswith('job-'):
            return
        # executor
        if podinfo['parent_uid'] != '':
            appuid = podinfo['parent_uid']
        # driver
        else:
            appuid = podinfo['pod_uid']
        # skip mongodb jobs which were already loaded from kubectl
        if podfrom == 'mongodb':
            if appuid in kubectl_jobs:
                return
        if appuid not in jobs_dict:
            jobs_dict[appuid] = {}
            jobs_dict[appuid]['pods'] = []
            # save jobs loaded from kubectl
            if podfrom == 'kubectl':
                kubectl_jobs.add(appuid)
        # job(same to driver) info
        if podinfo['parent_uid'] == '':
            jobs_dict[appuid]['spark_job_owner'] = podinfo['spark_job_owner']
            jobs_dict[appuid]['spark_job_id'] = podinfo['spark_job_id']
            jobs_dict[appuid]['namespace'] = podinfo['namespace']
            jobs_dict[appuid]['name'] = podinfo['podname']
            jobs_dict[appuid]['creation_timestamp'] = podinfo['creation_timestamp'].timestamp()
            jobs_dict[appuid]['phase'] = podinfo['phase']
        # driver & executor pod info
        jobs_dict[appuid]['pods'].append({
            'podname': podinfo['podname'],
            'phase': podinfo['phase'],
            'creation_timestamp': podinfo['creation_timestamp'].timestamp(),
            'duration_ns': podinfo['duration_ns']
        })

    # load jobs from kubectl api
    kubectljobs = kubectl.get_pods(namespace='default', tojson=True)
    if kubectljobs:
        for kubectljob in kubectljobs:
            save_pod_info(extract_pod_info(kubectljob), 'kubectl')

    # load jobs from mongodb
    mongojobs = Mongo().job_log_collection().find({}, {'logs': 0, '_id': 0})
    if mongojobs:
        for mongojob in mongojobs:
            save_pod_info(extract_pod_info(json.loads(mongojob['desc'])), 'mongodb')

    sorted_job_list = sorted(
        list(jobs_dict.items()),
        key=lambda x: x[1]['creation_timestamp'],
        reverse=True)
    return flask.render_template('jobs.html', jobs_list=sorted_job_list)


@app.route('/pod_describe/<path:pod_name>/<path:namespace>')
def pod_describe_hdl(pod_name, namespace='default'):
    """Handler of the pod info page"""
    mongodesc = Mongo().job_log_collection().find_one({'pod_name': pod_name,
                                                       'namespace': namespace},
                                                      {'desc': 1, '_id': 0})
    if mongodesc:
        mongodesctext = mongodesc['desc']
        return flask.render_template('pod_describe.html', pod_name=pod_name,
                                     pod_info=mongodesctext)
    return flask.render_template('pod_describe.html', pod_name=pod_name,
                                 pod_info=json.dumps(
                                     kubectl.describe_pod(pod_name, namespace, tojson=True),
                                     sort_keys=True, indent=4, separators=(', ', ': ')))


@app.route('/pod_delete', methods=['POST'])
def pod_delete_hdl():
    """Handler of the pod delete action"""
    pod_name = flask.request.form.get('pod_name', '')
    namespace = flask.request.form.get('namespace', '')
    if pod_name and namespace and pod_name.startswith('job-'):
        return str(kubectl.delete_pod(pod_name, namespace))
    else:
        return 'illegal pod name/namespace'


@app.route('/pod_log/<path:pod_name>/<path:namespace>')
def pod_log_hdl(pod_name, namespace='default'):
    """Handler of the pod log page"""
    return flask.render_template('pod_log.html', pod_name=pod_name, namespace=namespace)


@app.route('/pod_log_streaming/<path:pod_name>/<path:namespace>')
def pod_log_streaming_hdl(pod_name, namespace='default'):
    """Handler of the pod streaming log"""
    conv = Ansi2HTMLConverter()

    def decorate_logs(generator):
        """decorate logs"""
        for log in generator:
            log = log.decode('utf-8')
            yield conv.convert(ansi=log, full=False)

    mongologs = Mongo().job_log_collection().find_one({'pod_name': pod_name,
                                                       'namespace': namespace})
    if mongologs:
        mongologstext = mongologs['logs'] + '\ncurrent log was loaded from database.'
        return conv.convert(ansi=mongologstext, full=False)
    logs_generator = None
    try:
        logs_generator = kubectl.log_stream(pod_name, namespace)
    except Exception as ex:
        pass
    if logs_generator:
        return flask.Response(flask.stream_with_context(
            decorate_logs(logs_generator)), mimetype="text/plain")
    else:
        return 'pod does not exist or has been deleted'


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
    return flask.render_template('metrics.html', prefix=prefix, metrics=metrics)


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
