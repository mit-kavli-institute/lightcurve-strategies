lightcurve-strategies
=====================

.. image:: https://github.com/mit-kavli-institute/lightcurve-strategies/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/mit-kavli-institute/lightcurve-strategies/actions/workflows/ci.yml
   :alt: CI

.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://mit-kavli-institute.github.io/lightcurve-strategies/
   :alt: Documentation

**Hypothesis strategies for property-based testing with jaxoplanet.**

``lightcurve-strategies`` provides `Hypothesis <https://hypothesis.readthedocs.io/>`_
strategies that generate random but valid
`jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ objects —
transit orbits, Keplerian systems, starry surfaces, and surface systems —
for property-based testing of exoplanet light-curve code.

Quick start
-----------

.. code-block:: bash

   pip install lightcurve-strategies

.. code-block:: python

   from hypothesis import given
   from lightcurve_strategies import transit_orbits, light_curves, white_noise

   @given(transit_orbits())
   def test_orbit_is_valid(orbit):
       assert orbit.period > 0

Requirements
------------

- Python >= 3.11
- `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
- `Hypothesis <https://hypothesis.readthedocs.io/>`_
- `Astropy <https://www.astropy.org/>`_

License
-------

MIT
