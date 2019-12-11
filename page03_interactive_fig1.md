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
{% endfor %}


## Figure Description
Below is an example of an embedded interactive figure. It generates
two-dimensional random walks of 10,000 steps each time the button is clicked.
Moving the slider shows you in finer detail the position and history of the past
500 steps.

<!-- The below line includes the interactive figure. Do not change! -->
<center>

{% include_relative interactives/{{page.interactive}} %}

</center>


