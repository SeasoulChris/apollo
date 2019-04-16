<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
  {% block head %} {% endblock %}
</head>

<body>
<div class="container">
  <nav class="navbar navbar-default">
    <div class="container-fluid">
      <div class="navbar-header"><a class="navbar-brand" href="/">Apollo Data</a></div>
      <ul class="nav navbar-nav">
        <li><a href="{{ url_for('tasks_hdl', prefix='small-records') }}">Small Records</a></li>
        <li><a href="{{ url_for('tasks_hdl', prefix='public-test') }}">Public Test</a></li>
        <li><a href="{{ url_for('records_hdl') }}">Records</a></li>
        <li><a href="http://172.19.40.55:8000">Stale NFS Data</a></li>
      </ul>
    </div>
  </nav>

  {% for msg in get_flashed_messages() %}
  <div class="alert alert-warning">
    <button type="button" class="close" data-dismiss="alert">&times;</button>
    {{ msg }}
  </div>
  {% endfor %}

  {% block body %}
  {% endblock %}
</div>

<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</body>
</html>
