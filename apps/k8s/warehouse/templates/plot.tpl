{% block head %}

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>

{% endblock %}

{% block body %}

<div align="center">{{ data | plot_profiling }}</div>

{% endblock %}

