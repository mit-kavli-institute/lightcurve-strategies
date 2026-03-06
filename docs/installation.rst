Installation
============

``lightcurve-strategies`` is not yet published on PyPI.
Install from a local clone of the repository.

From source
-----------

.. code-block:: bash

   git clone https://github.com/<your-org>/lightcurve-strategies.git
   cd lightcurve-strategies
   pip install -e .

Development install
-------------------

To include linting, testing, and documentation tools:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Requirements
------------

- Python >= 3.11
- `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
- `Hypothesis <https://hypothesis.readthedocs.io/>`_
- `Astropy <https://www.astropy.org/>`_
