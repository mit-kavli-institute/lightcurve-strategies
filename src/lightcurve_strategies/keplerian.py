from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from jaxoplanet.orbits.keplerian import Body, Central, System

from lightcurve_strategies._units import strip_quantity


@st.composite
def centrals(
    draw: st.DrawFn,
    *,
    mass: SearchStrategy[float] | None = None,
    radius: SearchStrategy[float] | None = None,
) -> Central:
    """Generate random ``jaxoplanet.orbits.keplerian.Central`` instances.

    Each keyword accepts a Hypothesis strategy.  When ``None``, a sensible
    default strategy is used.  Strategies may yield ``astropy.units.Quantity``
    values — they will be converted to plain floats in the expected units
    (M_sun for mass, R_sun for radius).

    Parameters
    ----------
    mass:
        Strategy for stellar mass (M_sun).  Default: ``floats(0.1, 10.0)``.
    radius:
        Strategy for stellar radius (R_sun).  Default: ``floats(0.1, 10.0)``.
    """
    try:
        import astropy.units as u
    except ImportError:  # pragma: no cover
        u = None  # type: ignore[assignment]

    unit_msun = u.M_sun if u is not None else None
    unit_rsun = u.R_sun if u is not None else None

    if mass is None:
        mass = st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
    mass_val: float = strip_quantity(draw(mass), unit=unit_msun)

    if radius is None:
        radius = st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
    radius_val: float = strip_quantity(draw(radius), unit=unit_rsun)

    return Central(mass=mass_val, radius=radius_val)


@st.composite
def bodies(
    draw: st.DrawFn,
    *,
    period: SearchStrategy[float] | None = None,
    time_transit: SearchStrategy[float] | None = None,
    impact_param: SearchStrategy[float] | None = None,
    eccentricity: SearchStrategy[float] | None = None,
    omega_peri: SearchStrategy[float] | None = None,
    mass: SearchStrategy[float] | None = None,
    radius: SearchStrategy[float] | None = None,
) -> Body:
    """Generate random ``jaxoplanet.orbits.keplerian.Body`` instances.

    Circular orbits by default.  Providing ``eccentricity`` or ``omega_peri``
    enables eccentric orbits (the other gets a default strategy if not
    explicitly supplied).

    Parameters
    ----------
    period:
        Strategy for orbital period (days).  Default: ``floats(0.5, 365.0)``.
    time_transit:
        Strategy for mid-transit time (days).
        Default: ``floats(0.0, period)``.
    impact_param:
        Strategy for impact parameter (dimensionless, 0–1).
        Default: ``floats(0.0, 1.0)``.
    eccentricity:
        Strategy for eccentricity (0–0.9).  Default: ``None`` (circular).
    omega_peri:
        Strategy for argument of periastron (radians).
        Default: ``None`` (circular).
    mass:
        Strategy for body mass.  Default: ``None`` (omitted).
    radius:
        Strategy for body radius.  Default: ``None`` (omitted).
    """
    try:
        import astropy.units as u
    except ImportError:  # pragma: no cover
        u = None  # type: ignore[assignment]

    unit_day = u.day if u is not None else None

    # --- period ---
    if period is None:
        period = st.floats(0.5, 365.0, allow_nan=False, allow_infinity=False)
    period_val: float = strip_quantity(draw(period), unit=unit_day)

    # --- time_transit ---
    if time_transit is None:
        time_transit = st.floats(0.0, period_val, allow_nan=False, allow_infinity=False)
    time_transit_val: float = strip_quantity(draw(time_transit), unit=unit_day)

    # --- impact_param ---
    if impact_param is None:
        impact_param = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
    impact_param_val: float = strip_quantity(draw(impact_param))

    # --- eccentricity / omega_peri (linked pair) ---
    eccentric = eccentricity is not None or omega_peri is not None
    kwargs: dict = {}
    if eccentric:
        if eccentricity is None:
            eccentricity = st.floats(0.0, 0.9, allow_nan=False, allow_infinity=False)
        if omega_peri is None:
            omega_peri = st.floats(
                0.0, 6.283185307179586, allow_nan=False, allow_infinity=False
            )
        kwargs["eccentricity"] = strip_quantity(draw(eccentricity))
        kwargs["omega_peri"] = strip_quantity(draw(omega_peri))

    # --- optional mass/radius ---
    if mass is not None:
        kwargs["mass"] = strip_quantity(draw(mass))
    if radius is not None:
        kwargs["radius"] = strip_quantity(draw(radius))

    return Body(
        period=period_val,
        time_transit=time_transit_val,
        impact_param=impact_param_val,
        **kwargs,
    )


@st.composite
def systems(
    draw: st.DrawFn,
    *,
    central: SearchStrategy[Central] | None = None,
    body: SearchStrategy[Body] | None = None,
    min_bodies: int = 0,
    max_bodies: int = 5,
) -> System:
    """Generate random ``jaxoplanet.orbits.keplerian.System`` instances.

    Parameters
    ----------
    central:
        Strategy for the central star.  Default: ``centrals()``.
    body:
        Strategy for each body.  Default: ``bodies()``.
    min_bodies:
        Minimum number of bodies.  Default: ``0``.
    max_bodies:
        Maximum number of bodies.  Default: ``5``.
    """
    if central is None:
        central = centrals()
    if body is None:
        body = bodies()

    central_val: Central = draw(central)
    n_bodies: int = draw(st.integers(min_value=min_bodies, max_value=max_bodies))

    sys = System(central_val)
    for _ in range(n_bodies):
        sys = sys.add_body(body=draw(body))

    return sys
