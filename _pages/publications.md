---
layout: page
permalink: /publications/
title: publications
description: Please refer to my Google Scholar for a complete publication list.
years: [2026,2025,2021]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->

{% include bib_search.liquid %}

<div class="publications">

{%- for y in page.years %}

<h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
