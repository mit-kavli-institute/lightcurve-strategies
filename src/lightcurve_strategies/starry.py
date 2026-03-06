from __future__ import annotations

import math

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface

from lightcurve_strategies.keplerian import bodies, centrals


@st.composite
def surfaces(
    draw: st.DrawFn,
    *,
    inc: SearchStrategy[float] | None = None,
    obl: SearchStrategy[float] | None = None,
    period: SearchStrategy[float] | None = None,
    amplitude: SearchStrategy[float] | None = None,
    u: SearchStrategy[tuple] | None = None,
) -> Surface:
    """Generate random ``jaxoplanet.starry.Surface`` instances.

    Conservative defaults matching the ``Surface()`` constructor defaults.
    The ``y`` (Ylm) parameter is deliberately omitted — use
    ``st.builds(Surface, ...)`` for full control.

    Parameters
    ----------
    inc:
        Strategy for inclination (radians).  Default: ``just(π/2)``.
    obl:
        Strategy for obliquity (radians).  Default: ``just(None)``.
    period:
        Strategy for rotation period.  Default: ``just(None)``.
    amplitude:
        Strategy for luminosity amplitude.  Default: ``just(1.0)``.
    u:
        Strategy for limb-darkening coefficients tuple.  Default: ``just(())``.
    """
    if inc is None:
        inc = st.just(0.5 * math.pi)
    if obl is None:
        obl = st.just(None)
    if period is None:
        period = st.just(None)
    if amplitude is None:
        amplitude = st.just(1.0)
    if u is None:
        u = st.just(())

    inc_val = draw(inc)
    obl_val = draw(obl)
    period_val = draw(period)
    amplitude_val = draw(amplitude)
    u_val = draw(u)

    kwargs: dict = {}
    if inc_val is not None:
        kwargs["inc"] = inc_val
    if obl_val is not None:
        kwargs["obl"] = obl_val
    if period_val is not None:
        kwargs["period"] = period_val
    kwargs["amplitude"] = amplitude_val
    kwargs["u"] = u_val

    return Surface(**kwargs)


@st.composite
def surface_systems(
    draw: st.DrawFn,
    *,
    central: SearchStrategy[Central] | None = None,
    central_surface: SearchStrategy[Surface] | None = None,
    body: SearchStrategy[tuple[Body, Surface | None]] | None = None,
    min_bodies: int = 0,
    max_bodies: int = 5,
) -> SurfaceSystem:
    """Generate random ``jaxoplanet.starry.orbit.SurfaceSystem`` instances.

    Parameters
    ----------
    central:
        Strategy for the central star.  Default: ``centrals()``.
    central_surface:
        Strategy for the central surface.  Default: ``surfaces()``.
    body:
        Strategy producing ``(Body, Surface | None)`` tuples.
        Default: ``st.tuples(bodies(), surfaces())``.
    min_bodies:
        Minimum number of bodies.  Default: ``0``.
    max_bodies:
        Maximum number of bodies.  Default: ``5``.
    """
    if central is None:
        central = centrals()
    if central_surface is None:
        central_surface = surfaces()
    if body is None:
        body = st.tuples(bodies(), surfaces())

    central_val: Central = draw(central)
    central_surface_val: Surface = draw(central_surface)
    n_bodies: int = draw(st.integers(min_value=min_bodies, max_value=max_bodies))

    sys = SurfaceSystem(central=central_val, central_surface=central_surface_val)
    for _ in range(n_bodies):
        body_obj, surface_obj = draw(body)
        sys = sys.add_body(body=body_obj, surface=surface_obj)

    return sys
