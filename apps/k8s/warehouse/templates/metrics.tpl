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
@media (min-width: 1120px) {
   .modal-xl {
      width: 70%;
   }
}
.modal-dialog,
.modal-content {
    height: 90%;
}
.modal-body {
    max-height: calc(100% - 130px);
    overflow-y: scroll;
}
</style>


<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.2.0/socket.io.js"></script>
<script type="text/javascript" charset="utf-8">
  function attachFrame(key, serverAddr) {
    var src = serverAddr + '/plot_img/' + key;
    $('#theModal iframe').attr({'src': src});
    $('#theModalLabel').text(key);
  }


  function buildProfilingViewColumn(td, serverAddr, key, value) {
    if (!value.startsWith('[')) {
      td.appendChild(document.createTextNode(''));
      return
    }
    var link = document.createElement('a');
    link.innerHTML = 'View';
    link.href = '#';
    link.setAttribute('data-toggle', 'modal');
    link.setAttribute('data-target', '#theModal');
    link.setAttribute('data-key', key);
    link.setAttribute('data-serveraddr', serverAddr);
    link.addEventListener('click', function(e) {
        attachFrame(e.srcElement.getAttribute('data-key'), e.srcElement.getAttribute('data-serveraddr'));
      }, false);
    td.appendChild(link);
  }


  function buildTable(metrics, serverAddr) {
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
      var td_3 = document.createElement('td');
      buildProfilingViewColumn(td_3, serverAddr, keys[i], metrics[keys[i]]);
      tr.appendChild(td_3);
    }

    table.appendChild(tbody);
  }


  $(document).ready(function() {
    // TODO(Longtao): Fix the hardcoded URL later, which BTW is not working for socketio anyways
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
  
    buildTable(metrics, serverAddr);

    socket.on(connectChannel, function() {
      connected = true;
      socket.emit(clientRequestChannel, {'prefix': prefix});
    });

    setInterval(function() {
        if ($('#theModal').is(':visible')) {
          return;
        }
        if (connected) {
          socket.emit(clientRequestChannel, {'prefix': prefix});
        }
        else {
          // TODO(Longtao): remove this when socketio connection issue is fixed
          $.getJSON(serverAddr + metricsAjax, {'prefix': prefix}, function(serverMetrics) {
            buildTable(serverMetrics, serverAddr);
          });
        }
      }, timeInterval);

    socket.on(serverResponseChannel, function(serverMetrics) {
      buildTable(serverMetrics, serverAddr);
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


<div class="modal fade" id="theModal" tabindex="-1" role="dialog" aria-labelledby="theModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-xl" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h4 class="modal-title" id="theModalLabel">Modal Title</h4>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <iframe class="resp-iframe" frameborder="0" style="height:600px; width:100%; margin:0 auto;"></iframe>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary btn-sm" data-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>


{% endblock %}
