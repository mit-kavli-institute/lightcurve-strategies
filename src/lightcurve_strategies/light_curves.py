"""Hypothesis strategy for generating light curves with optional noise."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from jaxoplanet.starry.light_curves import light_curve
from jaxoplanet.starry.orbit import SurfaceSystem

from lightcurve_strategies.starry import surface_systems

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NoiseFn = Callable[[np.random.Generator, np.ndarray], np.ndarray]
"""Signature: (rng, time) -> noise array."""

KernelFn = Callable[[np.ndarray], np.ndarray]
"""Signature: (dt_matrix) -> covariance matrix."""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class LightCurveData(NamedTuple):
    """Container returned by :func:`light_curves`.

    Attributes
    ----------
    flux:
        Clean stellar flux, shape ``(n_times,)``.
    flux_with_noise:
        ``flux + noise``, shape ``(n_times,)``.
    noise:
        Noise realisation, shape ``(n_times,)``.
    system:
        The generated :class:`~jaxoplanet.starry.orbit.SurfaceSystem`.
    """

    flux: np.ndarray
    flux_with_noise: np.ndarray
    noise: np.ndarray
    system: SurfaceSystem


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def sq_exp_kernel(amplitude: float, length_scale: float) -> KernelFn:
    """Squared-exponential (RBF) covariance kernel.

    Parameters
    ----------
    amplitude:
        Signal amplitude *a*.
    length_scale:
        Characteristic length scale *ℓ*.

    Returns
    -------
    KernelFn:
        ``K(dt) = a² exp(-dt² / 2ℓ²)``
    """

    def _kernel(dt: np.ndarray) -> np.ndarray:
        return amplitude**2 * np.exp(-(dt**2) / (2.0 * length_scale**2))

    return _kernel


def matern32_kernel(amplitude: float, length_scale: float) -> KernelFn:
    """Matérn-3/2 covariance kernel.

    Parameters
    ----------
    amplitude:
        Signal amplitude *a*.
    length_scale:
        Characteristic length scale *ℓ*.

    Returns
    -------
    KernelFn:
        ``K(dt) = a² (1 + √3|dt|/ℓ) exp(-√3|dt|/ℓ)``
    """

    def _kernel(dt: np.ndarray) -> np.ndarray:
        r = np.sqrt(3.0) * np.abs(dt) / length_scale
        return np.asarray(amplitude**2 * (1.0 + r) * np.exp(-r))

    return _kernel


# ---------------------------------------------------------------------------
# Noise factories
# ---------------------------------------------------------------------------


def white_noise(scale: float) -> NoiseFn:
    """Independent Gaussian noise.

    Parameters
    ----------
    scale:
        Standard deviation of the noise.

    Returns
    -------
    NoiseFn:
        ``noise(rng, time) -> rng.normal(0, scale, len(time))``
    """

    def _noise(rng: np.random.Generator, time: np.ndarray) -> np.ndarray:
        return np.asarray(rng.normal(0.0, scale, size=time.shape[0]))

    return _noise


def red_noise(kernel: KernelFn, jitter: float = 0.0) -> NoiseFn:
    """Correlated (GP-like) noise via Cholesky decomposition.

    Parameters
    ----------
    kernel:
        Covariance kernel function (e.g. from :func:`sq_exp_kernel`).
    jitter:
        Diagonal regularisation added to the covariance matrix.

    Returns
    -------
    NoiseFn:
        Draws correlated samples ``L @ z`` where ``L = cholesky(K + jitter*I)``.
    """

    def _noise(rng: np.random.Generator, time: np.ndarray) -> np.ndarray:
        dt = time[:, None] - time[None, :]
        cov = kernel(dt) + jitter * np.eye(len(time))
        chol = np.linalg.cholesky(cov)
        return chol @ rng.standard_normal(len(time))

    return _noise


def combined_noise(*noise_fns: NoiseFn) -> NoiseFn:
    """Additive combination of multiple noise sources.

    Parameters
    ----------
    *noise_fns:
        One or more :data:`NoiseFn` callables.

    Returns
    -------
    NoiseFn:
        Sums the output of every constituent noise function.
    """

    def _noise(rng: np.random.Generator, time: np.ndarray) -> np.ndarray:
        total = np.zeros(len(time))
        for fn in noise_fns:
            total = total + fn(rng, time)
        return total

    return _noise


# ---------------------------------------------------------------------------
# Composite strategy
# ---------------------------------------------------------------------------


@st.composite
def light_curves(
    draw: st.DrawFn,
    *,
    time: np.ndarray,
    system: SearchStrategy[SurfaceSystem] | None = None,
    order: int = 10,
    noise: SearchStrategy[NoiseFn] | None = None,
    seed: SearchStrategy[int] | None = None,
) -> LightCurveData:
    """Generate a light curve with optional noise.

    Parameters
    ----------
    time:
        Time grid in days.
    system:
        Strategy for the stellar system.
        Default: ``surface_systems(min_bodies=1, max_bodies=1)``.
    order:
        Spherical harmonic expansion order for light-curve computation.
    noise:
        Strategy that draws a :data:`NoiseFn`.  When ``None`` (default),
        no noise is added.
    seed:
        Strategy for the RNG seed.  Default: ``st.integers(0, 2**32 - 1)``.
        Only drawn when *noise* is not ``None``.
    """
    if system is None:
        system = surface_systems(min_bodies=1, max_bodies=1)

    system_val: SurfaceSystem = draw(system)

    clean_flux: np.ndarray = np.asarray(
        light_curve(system_val, order=order)(jnp.asarray(time))[:, 0]
    )

    if noise is not None:
        noise_fn: NoiseFn = draw(noise)
        if seed is None:
            seed = st.integers(0, 2**32 - 1)
        seed_val: int = draw(seed)
        rng = np.random.default_rng(seed_val)
        noise_array: np.ndarray = noise_fn(rng, time)
    else:
        noise_array = np.zeros_like(clean_flux)

    return LightCurveData(
        flux=clean_flux,
        flux_with_noise=clean_flux + noise_array,
        noise=noise_array,
        system=system_val,
    )
