---
title: 'portrait: observation metrics for periodic events'
tags:
  - Python
  - astronomy
  - exoplanets
  - periods
authors:
  - name: Lionel Garcia
    orcid: 0000-0002-4296-2246
    equal-contrib: true
    affiliation: 1
  - name: Elsa Ducrot
    orcid: 0000-0002-7008-6888
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, USA
   index: 1
 - name: LESIA, Observatoire de Paris, CNRS, Université Paris Diderot, Université Pierre et Marie Curie, Meudon, France
   index: 2
date: 22 May 2024
bibliography: paper.bib
---

# Summary

The observation of a rotating object is often constrained by its visibility, leading to sparse measurement time series. This is the case, for example, when measuring the light of a rotating star from the ground, which can only be done at night, when the star is visible in the sky, or with the right atmospheric conditions. If the goal is to observe a full rotation of the object, it is important to quantify how much phase has been covered by a given set of observations. In this context, the `portrait` package provides tools to compute and visualize observation metrics for periodic events.

# Statement of need

`portrait` is a Python package designed to provide observation metrics of periodic events. In Astronomy, it has been used in the context of transiting exoplanets detections (e.g., @Delrez2022; @Schanche2022), to find out how much of an orbit with a given period $P$ has been covered by a set of observations. This is particularly important from the ground, where experimental constraints limit the possible times of observations, or to estimate the completeness of a transit survey.

At scale, `portrait` has been used to design the SPECULOOS transit survey, optimizing the observing strategy that maximizes the coverage of transiting exoplanets orbiting in the habitable zone of their star [@Sebastian2021]. The package is designed to be user-friendly, with a focus on simplicity and ease of use. It is highly optimized and hardware-accelerated thanks to JAX [@jax2018github], which allows for fast computation of the metrics on large datasets.


## Acknowledgements
`portrait` dependencies include numpy [@harris2020array], jax and jaxlib [@jax2018github]

# References
