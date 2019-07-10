{% extends "base.tpl" %}

{% block head %}

<title>Apollo Fuel - Record - {{ record.path }}</title>

<style>
.green {color: green;}
.red {color: red;}
.text_center {text-align: center;}
</style>
{% endblock %}

{% block body %}

<div class="panel panel-default">
  <div class="panel-heading">Information</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Mode</th>
          <th>Map</th>
          <th>Vehicle</th>
          <th>Begin Time</th>
          <th>Duration</th>
          <th>Size</th>
          <th>Mileage (Auto/Total)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{{ record.hmi_status.current_mode }}</td>
          <td>{{ record.hmi_status.current_map }}</td>
          <td>{{ record.hmi_status.current_vehicle }}</td>
          <td>{{ record.header.begin_time | timestamp_ns_to_time }}</td>
          <td>{{ ((record.header.end_time - record.header.begin_time) / 1000000000.0) | round(1) }} s</td>
          <td>{{ record.header.size | readable_data_size }}</td>
          <td>{{ record.stat.mileages['COMPLETE_AUTO_DRIVE'] | meter_to_miles }} /
              {{ record.stat.mileages.values() | sum | meter_to_miles }} miles</td>
        </tr>
      </tbody>
    </table>

    {# Draw map path. #}
    {% if record.stat.driving_path %}
      <script type="text/javascript" src="http://maps.google.com/maps/api/js?sensor=false&key=AIzaSyDZsO7KfO7mfE9lIkInRxwQgn-1qufqww0"></script>
      <script type="text/javascript" src="{{ url_for('static', filename='js/gmap_util.js') }}"></script>
      <div style="width:100%; height:350px;">
        <div id="gmap_canvas" style="width: 100%; height: 100%;"></div>
        <script>
          {{ record.stat.driving_path | draw_path_on_gmap('gmap_canvas') }}
          {{ record | draw_disengagements_on_gmap }}
        </script>
      </div>

      <table class="table text_center">
        <tbody>
          <tr>
            <td><span class="glyphicon glyphicon-record green"></span> Start Point</td>
            <td><span class="glyphicon glyphicon-record red"></span> Stop Point</td>
            <td><span class="glyphicon glyphicon-map-marker red"></span> Disengagement</td>
          </tr>
        </tbody>
      </table>
    {% endif %}
  </div>
</div>

{% if sub_records %}
<div class="panel panel-default">
  <div class="panel-heading">Records</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Begin Time</th>
          <th>Duration</th>
          <th>Size</th>
          <th>Path</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for record in sub_records %}
          <tr>
            <td>{{ record.header.begin_time | timestamp_ns_to_time }}</td>
            <td>{{ ((record.header.end_time - record.header.begin_time) / 1000000000.0) | round(1) }} s</td>
            <td>{{ record.header.size | readable_data_size }}</td>
            <td>{{ record.path }}</td>
            <td><a href="{{ url_for('record_hdl', record_path=record.path[1:]) }}">View</a></td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

{% if record.disengagements %}
<div class="panel panel-default">
  <div class="panel-heading">Disengagements</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Time</th>
          <th>Description</th>
          <th>Offset</th>
        </tr>
      </thead>
      <tbody>
        {% for dis in record.disengagements %}
          <tr>
            <td>{{ dis.time | timestamp_to_time }}</td>
            <td>{{ dis.desc }}</td>
            <td>{{ (dis.time - record.header.begin_time / 1000000000.0) | round(1) }} s</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

{% if record.drive_events %}
<div class="panel panel-default">
  <div class="panel-heading">Drive Events</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Time</th>
          <th>Type</th>
          <th>Description</th>
          <th>Offset</th>
        </tr>
      </thead>
      <tbody>
        {% for event in record.drive_events %}
          <tr>
            <td>{{ event.header.timestamp_sec | timestamp_to_time }}</td>
            <td>{% for type in event.type %} {{ type | drive_event_type_name }} {% endfor %}</td>
            <td>{{ event.event }}</td>
            <td>{{ (event.header.timestamp_sec - record.header.begin_time / 1000000000.0) | round(1) }} s</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

{% if record.stat.planning_stat and record.stat.planning_stat.latency%}
<div class="panel panel-default">
  <div class="panel-heading">Planning Latency</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Metrics</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Max</td>
          <td>{{ record.stat.planning_stat.latency.max }}</td>
        </tr>
        <tr>
          <td>Min</td>
          <td>{{ record.stat.planning_stat.latency.min }}</td>
        </tr>
        <tr>
          <td>Avg</td>
          <td>{{ record.stat.planning_stat.latency.avg }}</td>
        </tr>
        <tr>
          <td>Latency 0-10 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_0_10_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 20-40 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_20_40_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 40-60 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_40_60_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 60-80 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_60_80_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 80-100 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_80_100_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 100-120 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_100_120_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 120-150 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_120_150_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 150-200 ms</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_150_200_ms"] }}</td>
        </tr>
        <tr>
          <td>Latency 200 ms and up</td>
          <td>{{ record.stat.planning_stat.latency.latency_hist["latency_200_up_ms"] }}</td>
        </tr>
      </tbody>
    </table>
    <div>{{ record | plot_record }}</div>
  </div>
</div>
{% endif %}

<div class="panel panel-default">
  <div class="panel-heading">Channels</div>
  <div class="panel-body">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Name</th>
          <th>Messages</th>
        </tr>
      </thead>
      <tbody>
        {% for channel, msg_count in record.channels.items() %}
          <tr>
            <td>{{ channel }}</td>
            <td>{{ msg_count }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

{% endblock %}
