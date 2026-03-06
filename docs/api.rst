API Reference
=============

All strategies are importable directly from the top-level package:

.. code-block:: python

   from lightcurve_strategies import transit_orbits, centrals, bodies, systems
   from lightcurve_strategies import surfaces, surface_systems
   from lightcurve_strategies import light_curves, white_noise, red_noise

Transit
-------

.. autofunction:: lightcurve_strategies.transit_orbits

Keplerian
---------

.. autofunction:: lightcurve_strategies.centrals

.. autofunction:: lightcurve_strategies.bodies

.. autofunction:: lightcurve_strategies.systems

Starry
------

.. autofunction:: lightcurve_strategies.surfaces

.. autofunction:: lightcurve_strategies.surface_systems

Light Curves
------------

.. autoclass:: lightcurve_strategies.LightCurveData

.. autofunction:: lightcurve_strategies.light_curves

.. autofunction:: lightcurve_strategies.white_noise

.. autofunction:: lightcurve_strategies.red_noise

.. autofunction:: lightcurve_strategies.combined_noise

.. autofunction:: lightcurve_strategies.sq_exp_kernel

.. autofunction:: lightcurve_strategies.matern32_kernel
