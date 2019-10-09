{% extends "base.tpl" %}

{% block head %}

<title>Apollo Fuel - Metrics</title>


<style>
.topic_options {
  min-width: 500px;
  font-size: 16px;
}
.page_button {
  width: 40px;
}
input[type=text] {
  width: 130px;
  height: 40px;
  box-sizing: border-box;
  border: 2px solid #ccc;
  border-radius: 5px;
  background-color: white;
  background-position: 10px 10px;
  background-repeat: no-repeat;
  padding: 12px 20px 12px 40px;
  -webkit-transition: width 0.4s ease-in-out;
  transition: width 0.4s ease-in-out;
}
input[type=text]:focus {
  width: 100%;
}
</style>


<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.2.0/socket.io.js"></script>
<script type="text/javascript" charset="utf-8">
  function buildTable(metrics) {
    var keys = Object.keys(metrics);
    var table = document.getElementById('metricstable');

    var tbodys = table.getElementsByTagName('tbody');
    if (tbodys.length > 0) {
      table.removeChild(tbodys[0]);
    }
    var tbody = document.createElement('tbody');

    for (var i = 0; i < keys.length; i++) {
      var tr = document.createElement('tr');
      var td_1 = document.createElement('td');
      td_1.appendChild(document.createTextNode(keys[i]));
      tr.appendChild(td_1);
      var td_2 = document.createElement('td');
      td_2.appendChild(document.createTextNode(metrics[keys[i]]));
      tr.appendChild(td_2);
      tbody.appendChild(tr);
    }

    table.appendChild(tbody);
  }

  $(document).ready(function() {
    // TODO(Longtao): Fix the Hardcoded URL later, which BTW is not working for socketio anyways
    var serverAddr = 'http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:warehouse-service:8000/proxy';
    var socket = io.connect(serverAddr);
    var connected = false;
    var timeInterval = 5000;
    var connectChannel = 'connect';
    var clientRequestChannel = 'client_request_metrics_event';
    var serverResponseChannel = 'server_response_metrics';
    var metricsAjax = '/metrics_ajax';

    var metrics = {{ metrics | tojson | safe }};
    var prefix = '{{ prefix }}';
  
    buildTable(metrics);

    socket.on(connectChannel, function() {
      connected = true;
      socket.emit(clientRequestChannel, {'prefix': prefix});
    });

    setInterval(function() {
        if (connected) {
          socket.emit(clientRequestChannel, {'prefix': prefix});
        }
        else {
          // TODO(Longtao): remove this when socketio connection issue is fixed
          $.getJSON(serverAddr + metricsAjax, {'prefix': prefix}, function(serverMetrics) {
            buildTable(serverMetrics);
          });
        }
      }, timeInterval);

    socket.on(serverResponseChannel, function(serverMetrics) {
      buildTable(serverMetrics);
    });

  });
</script>



{% endblock %}

{% block body %}

<div>
  <form action="{{ url_for('metrics_hdl') }}" method="get">
    <input type="text" id="prefix" name="prefix" placeholder=" {% if prefix %} {{ prefix }}... {% else %} Prefix... {% endif %} ">
  </form>

  <br>

  <div class="panel panel-default">
    <div class="panel-heading">Metrics</div>
    <div class="panel-body">
      <table class="table" id="metricstable">
        <thead>
          <tr>
            <th>Key</th>
            <th>Value</th>
          </tr>
        </thead>
      </table>
    </div>
  </div>

</div>

{% endblock %}
