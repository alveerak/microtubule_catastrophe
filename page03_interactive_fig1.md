---
layout: page
title: Analysis
permalink: interactive_a
sidebar: true
interactive: interactive_1.html
---
---



{% for entry in site.data.analysis %}

{% if entry[0] != 'title' %}
{% if entry[0] != 'authors' %}
## {{entry[0]}}
{{entry[1]}}
{% endif %}
{% endif %}

{% if entry[0] == 'Generating ECDFs' %}
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/ECDF_plots.png" width = "500" ></center>
{% endif %}

{% if entry[0] == 'Simulating Microtubule Catastrophe' %}
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/simulations.png" width = "500" ></center>
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/compare_theory_vs_sim.png" width = "500" ></center>
{% endif %}

{% if entry[0] == 'Confidence Intervals for Microtubule Catastrophe' %}
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/ECDF_w_ConfInt.png" width = "500" ></center>
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/upper_lower_bounds.png" width = "500" ></center>
{% endif %}

{% if entry[0] == 'Comparison of Models' %}
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/jitter.png" width = "500" ></center>
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/ECDFs_conc.png" width = "500" ></center>
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/MLEs.png" width = "260" ></center>
<center> <img src="{{site.url}}/{{site.baseurl}}/assets/img/conf_int.png" width = "500" ></center>
{% endif %}

{% endfor %}

