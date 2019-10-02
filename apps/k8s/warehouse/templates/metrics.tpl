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

{% endblock %}

{% block body %}

<div>
  <form action="{{ url_for('metrics_hdl') }}" method="post">
    <input type="text" name="prefix" placeholder="Prefix...">
  </form>

  <br>

  <div class="panel panel-default">
    <div class="panel-heading">Metrics</div>
    <div class="panel-body">
      <table class="table table-striped">
        <thead>
          <tr>
            <th>Keys</th>
            <th>Values</th>
          </tr>
        </thead>
        <tbody>
          {% for key in metrics %}
            <tr>
              <td>{{ key }}</td>
              <td>{{ metrics[key] }}</td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>

{% endblock %}
