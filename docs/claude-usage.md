# Using lightcurve-strategies (Claude Code reference)

> Copy the relevant sections below into your project's CLAUDE.md so Claude Code
> knows how to generate test light curves without reading source code.

## What this package does

`lightcurve-strategies` provides Hypothesis strategies that generate random but
physically valid jaxoplanet objects — transit orbits, Keplerian systems, starry
surfaces, and full light curves with noise — for property-based testing of
exoplanet light-curve analysis code.

## Install

```bash
pip install lightcurve-strategies
```

## Environment

Always set `JAX_PLATFORMS=cpu` before importing (jaxoplanet uses JAX):

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
```

Or in CI: `env: JAX_PLATFORMS: cpu`.

## Public API

Everything is importable from the top-level package:

```python
from lightcurve_strategies import (
    # Orbital strategies
    transit_orbits,       # -> TransitOrbit
    centrals,             # -> Central  (star)
    bodies,               # -> Body     (planet)
    systems,              # -> System   (Keplerian)

    # Starry strategies
    surfaces,             # -> Surface  (with optional Ylm maps)
    surface_systems,      # -> SurfaceSystem

    # End-to-end light curve
    light_curves,         # -> LightCurveData (flux, flux_with_noise, noise, system)
    LightCurveData,       # NamedTuple return type

    # Noise factories
    white_noise,          # independent Gaussian
    red_noise,            # GP-correlated (Cholesky)
    combined_noise,       # sum of multiple noise sources

    # Kernel functions (for red_noise)
    sq_exp_kernel,        # squared-exponential / RBF
    matern32_kernel,      # Matérn-3/2
)
```

## Key pattern: `st.just()` to pin parameters

Every strategy parameter accepts either `None` (use sensible random default) or
a Hypothesis strategy. Use `st.just(value)` to fix a parameter to a specific value:

```python
from hypothesis import strategies as st

# Fully random star
star = centrals()

# Sun-like star (fixed mass and radius)
star = centrals(mass=st.just(1.0), radius=st.just(1.0))

# Random mass, fixed radius
star = centrals(radius=st.just(1.0))
```

This pattern applies to **all** strategies in the package.

## Common recipes

### 1. Simple transit orbit

```python
from hypothesis import given
from lightcurve_strategies import transit_orbits

@given(orbit=transit_orbits())
def test_orbit_has_positive_period(orbit):
    assert orbit.period > 0
```

Parameters: `period` (days), `duration` (days), `time_transit` (days),
`impact_param` (0–1), `radius_ratio` (dimensionless).

### 2. Keplerian system with one planet

```python
from hypothesis import given, strategies as st
from lightcurve_strategies import centrals, bodies, systems

@given(
    sys=systems(
        central=centrals(mass=st.just(1.0), radius=st.just(1.0)),
        body=bodies(period=st.just(3.0), radius=st.just(0.1)),
        min_bodies=1,
        max_bodies=1,
    )
)
def test_system(sys):
    ...
```

### 3. Full light curve (the most common use case)

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from lightcurve_strategies import (
    centrals, bodies, surfaces, surface_systems,
    light_curves, white_noise,
)

time = np.linspace(-0.2, 0.2, 500)

system_strategy = surface_systems(
    central=centrals(mass=st.just(1.0), radius=st.just(1.0)),
    central_surface=surfaces(u=st.just((0.1, 0.3))),  # limb darkening
    body=st.tuples(
        bodies(period=st.just(3.0), radius=st.just(0.1)),
        surfaces(),
    ),
    min_bodies=1,
    max_bodies=1,
)

@given(
    data=light_curves(
        time=time,
        system=system_strategy,
        noise=st.just(white_noise(scale=5e-4)),
    )
)
@settings(max_examples=50)
def test_transit_dips(data):
    # data.flux            — clean transit (np.ndarray)
    # data.flux_with_noise — transit + noise (np.ndarray)
    # data.noise           — noise realisation (np.ndarray)
    # data.system          — the SurfaceSystem instance
    baseline = data.flux[0]
    assert np.all(data.flux <= baseline + 1e-6)
```

### 4. Red (correlated) noise

```python
from lightcurve_strategies import red_noise, sq_exp_kernel, matern32_kernel

# Squared-exponential kernel
noise_fn = red_noise(
    kernel=sq_exp_kernel(amplitude=3e-4, length_scale=0.05),
    jitter=1e-10,
)

# Matérn-3/2 kernel (rougher, more realistic)
noise_fn = red_noise(
    kernel=matern32_kernel(amplitude=3e-4, length_scale=0.05),
    jitter=1e-10,
)
```

### 5. Combined noise (white + red)

```python
from lightcurve_strategies import combined_noise, white_noise, red_noise, sq_exp_kernel

noise_fn = combined_noise(
    white_noise(scale=3e-4),
    red_noise(kernel=sq_exp_kernel(3e-4, 0.05), jitter=1e-10),
)
```

### 6. Spotted stellar surface (spherical harmonics)

```python
import numpy as np
from hypothesis import strategies as st
from jaxoplanet.starry.ylm import Ylm
from lightcurve_strategies import surfaces

np.random.seed(42)
y = Ylm.from_dense([1.00, *np.random.normal(0.0, 2e-2, size=15)])

spotted = surfaces(
    y=st.just(y),
    inc=st.just(1.0),
    obl=st.just(0.2),
    period=st.just(27.0),
    u=st.just((0.1, 0.1)),
)
```

### 7. Deterministic seeds for reproducible noise

```python
strategy = light_curves(
    time=time,
    system=system_strategy,
    noise=st.just(white_noise(scale=1e-3)),
    seed=st.just(42),  # same noise every time
)
```

When `seed` is omitted, Hypothesis controls it (deterministic within a test run,
shrinkable on failure).

## Strategy parameter reference

| Strategy | Parameter | Unit / range | Default |
|----------|-----------|-------------|---------|
| `transit_orbits()` | `period` | days | `floats(0.5, 365)` |
| | `duration` | days | `floats(0.001, min(0.5*P, 2))` |
| | `time_transit` | days | `floats(0, P)` |
| | `impact_param` | 0–1 | `floats(0, 1)` |
| | `radius_ratio` | dimensionless | `floats(0.001, 0.3)` |
| `centrals()` | `mass` | M_sun | `floats(0.1, 10)` |
| | `radius` | R_sun | `floats(0.1, 10)` |
| `bodies()` | `period` | days | `floats(0.5, 365)` |
| | `time_transit` | days | `floats(0, P)` |
| | `impact_param` | 0–1 | `floats(0, 1)` |
| | `eccentricity` | 0–0.9 | `None` (circular) |
| | `omega_peri` | radians | `None` (circular) |
| | `mass` | M_sun | `None` (omitted) |
| | `radius` | R_sun | `None` (omitted) |
| `surfaces()` | `y` | Ylm | `just(None)` (uniform) |
| | `inc` | radians | `just(π/2)` |
| | `obl` | radians | `just(None)` |
| | `period` | rotation period | `just(None)` |
| | `amplitude` | dimensionless | `just(1.0)` |
| | `u` | tuple | `just(())` |
| `surface_systems()` | `central` | Central strategy | `centrals()` |
| | `central_surface` | Surface strategy | `surfaces()` |
| | `body` | (Body, Surface) strategy | `tuples(bodies(), surfaces())` |
| | `min_bodies` | int | `0` |
| | `max_bodies` | int | `5` |
| `light_curves()` | `time` | np.ndarray (days) | **required** |
| | `system` | SurfaceSystem strategy | `surface_systems(min=1, max=1)` |
| | `order` | int | `10` |
| | `noise` | NoiseFn strategy | `None` (no noise) |
| | `seed` | int strategy | `integers(0, 2^32-1)` |

## Gotchas

- **`JAX_PLATFORMS=cpu`** must be set before importing. Without it, JAX may fail
  or try to use a GPU.
- **`st.just()` wrapping** — pass strategies, not raw values. `bodies(period=3.0)`
  is wrong; use `bodies(period=st.just(3.0))`.
- **`surface_systems` body parameter** takes a strategy of `(Body, Surface)` tuples,
  not separate body/surface strategies. Use `st.tuples(bodies(...), surfaces(...))`.
- **`light_curves` returns `LightCurveData`** (a NamedTuple), not a raw array.
  Access `.flux`, `.flux_with_noise`, `.noise`, `.system`.
- **Astropy units** are accepted by `transit_orbits`, `centrals`, `bodies` — they
  auto-strip to the expected unit. `surfaces` does **not** do unit conversion.
- **`order` parameter** in `light_curves()` controls spherical harmonic expansion
  accuracy. Default `10` is fine for most tests. Lower for speed, higher for precision.
