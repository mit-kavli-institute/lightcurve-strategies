from __future__ import annotations


def strip_quantity(value: float | int | object, unit: object = None) -> float:
    """Extract a plain float from a value that may be an astropy Quantity.

    Parameters
    ----------
    value:
        A plain number or an ``astropy.units.Quantity``.
    unit:
        If provided and *value* is a Quantity, convert to this unit first.

    Returns
    -------
    float
        The numeric value as a plain Python float.
    """
    try:
        from astropy.units import Quantity
    except ImportError:  # pragma: no cover
        return float(value)  # type: ignore[arg-type]

    if isinstance(value, Quantity):
        if unit is not None:
            return float(value.to(unit).value)
        return float(value.value)

    return float(value)  # type: ignore[arg-type]
