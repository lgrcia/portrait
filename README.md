# portrait

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/logo@3x.png" width="200">
</p>

<p align="center">
  Observation metrics for periodic events
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/portrait">
      <img src="https://img.shields.io/badge/github-lgrcia/portrait-blue.svg?style=flat" alt="github"/>
    </a>
    <a href="">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
    <a href="https://portrait.readthedocs.io">
      <img src="https://img.shields.io/badge/documentation-black.svg?style=flat" alt="documentation"/>
    </a>
  </p>
</p>

A Python package to compute and visualize observation metrics for periodic events.

In Astronomy, *portrait* is useful to answer the following questions:
- How much of an orbit with period $P$ has been observed given certain observation times? ([tutorial](docs/ipynb/coverage))
- What period $P$ matches with this list of events? ([tutorial](docs/ipynb/periodmatch))
- How a certain target must be observed to cover all orbits with periods lower than $P$ days? 