---
layout: archive
title: "Birds"
permalink: /birds/
author_profile: true
entries_layout: grid
evolve-photo:
- aspect: "1.3333"
  url: "eastern_bluebird.jpg"
  image_path: "/images/eastern_bluebird.jpg"
  alt: "Eastern bluebird"
- aspect: "0.75"
  url: "annas_hummingbird.jpg"
  image_path: "/images/annas_hummingbird.jpg"
  alt: "Anna's hummingbird here"
  end_row: "true"
---



{% include base_path %}


{% for post in site.portfolio %}
{% include archive-single.html %}
{% endfor %}

{% include flexgallery id="evolve-photo" caption="Evolving photography - using a compact digital camera, a digital SLR and then a smartphone camera (Pictures courtesy of Aravind Iyer)"%}