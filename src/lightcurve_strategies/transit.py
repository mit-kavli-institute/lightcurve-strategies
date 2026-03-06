from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from jaxoplanet.orbits import TransitOrbit

from lightcurve_strategies._units import strip_quantity


@st.composite
def transit_orbits(
    draw: st.DrawFn,
    *,
    period: SearchStrategy[float] | None = None,
    duration: SearchStrategy[float] | None = None,
    time_transit: SearchStrategy[float] | None = None,
    impact_param: SearchStrategy[float] | None = None,
    radius_ratio: SearchStrategy[float] | None = None,
) -> TransitOrbit:
    """Generate random ``jaxoplanet.orbits.TransitOrbit`` instances.

    Each keyword accepts a Hypothesis strategy.  When ``None``, a sensible
    default strategy is used.  Strategies may yield ``astropy.units.Quantity``
    values — they will be converted to plain floats in the expected units
    (days for time parameters, dimensionless for ratios).

    Parameters
    ----------
    period:
        Strategy for orbital period (days).  Default: ``floats(0.5, 365.0)``.
    duration:
        Strategy for transit duration (days).
        Default: ``floats(0.001, min(0.5 * period, 2.0))``.
    time_transit:
        Strategy for mid-transit time (days).
        Default: ``floats(0.0, period)``.
    impact_param:
        Strategy for impact parameter (dimensionless, 0–1).
        Default: ``floats(0.0, 1.0)``.
    radius_ratio:
        Strategy for planet-to-star radius ratio (dimensionless).
        Default: ``floats(0.001, 0.3)``.
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

    # --- duration (must be < period) ---
    if duration is None:
        max_dur = min(0.5 * period_val, 2.0)
        duration = st.floats(0.001, max_dur, allow_nan=False, allow_infinity=False)
    duration_val: float = strip_quantity(draw(duration), unit=unit_day)

    # --- time_transit ---
    if time_transit is None:
        time_transit = st.floats(0.0, period_val, allow_nan=False, allow_infinity=False)
    time_transit_val: float = strip_quantity(draw(time_transit), unit=unit_day)

    # --- impact_param ---
    if impact_param is None:
        impact_param = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
    impact_param_val: float = strip_quantity(draw(impact_param))

    # --- radius_ratio ---
    if radius_ratio is None:
        radius_ratio = st.floats(0.001, 0.3, allow_nan=False, allow_infinity=False)
    radius_ratio_val: float = strip_quantity(draw(radius_ratio))

    return TransitOrbit(
        period=period_val,
        duration=duration_val,
        time_transit=time_transit_val,
        impact_param=impact_param_val,
        radius_ratio=radius_ratio_val,
    )
